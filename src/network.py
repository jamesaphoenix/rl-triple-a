"""Actor-Critic networks for TripleA RL.

Two architectures:
- ActorCriticV2: Original flat MLP (8.9M params, 16k obs) — kept for compatibility
- ActorCriticV3: Territory-aware architecture with:
    - Compressed observations (friendly/enemy sums, strategic features)
    - GNN message passing along territory adjacency graph
    - Multi-head attention over territories
    - Skip connections + deeper network
    - Separate per-territory action heads
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ── V2: Original flat MLP (kept for checkpoint compatibility) ────────

class ActorCriticV2(nn.Module):
    """Original flat MLP. 8.9M params on 16,215-dim input."""

    def __init__(self, obs_size: int, action_dim: int, hidden_size: int = 512):
        super().__init__()
        self.action_dim = action_dim
        self.shared = nn.Sequential(
            nn.Linear(obs_size, hidden_size), nn.LayerNorm(hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.LayerNorm(hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, 256), nn.LayerNorm(256), nn.ReLU(),
        )
        self.policy_mean = nn.Sequential(
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, action_dim), nn.Sigmoid(),
        )
        self.log_std = nn.Parameter(torch.zeros(action_dim) - 0.5)
        self.value_head = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1),
        )

    def forward(self, obs):
        features = self.shared(obs)
        return self.policy_mean(features), self.value_head(features)

    def get_action(self, obs, deterministic=False):
        action_mean, value = self.forward(obs)
        std = torch.exp(self.log_std).expand_as(action_mean)
        if deterministic:
            action = action_mean
        else:
            dist = torch.distributions.Normal(action_mean, std)
            action = dist.sample().clamp(0.0, 1.0)
        dist = torch.distributions.Normal(action_mean, std)
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action.squeeze(0).detach().cpu().numpy(), log_prob.squeeze(0), value.squeeze(-1).squeeze(0)

    def evaluate_actions(self, obs, actions):
        action_mean, value = self.forward(obs)
        std = torch.exp(self.log_std).expand_as(action_mean)
        dist = torch.distributions.Normal(action_mean, std)
        return dist.log_prob(actions).sum(dim=-1), value.squeeze(-1), dist.entropy().sum(dim=-1)


# ── V3: Territory-Aware GNN + Attention Architecture ─────────────────

NUM_TERRITORIES = 162
NUM_PLAYERS = 7
NUM_UNIT_TYPES = 13


class TerritoryEncoder(nn.Module):
    """Encode per-territory features from raw observation.

    Takes the flat 16,215-dim obs and reshapes it into per-territory features,
    compressing 7-player unit counts into friendly/enemy aggregates.

    Input: (batch, 16215)
    Output: (batch, num_territories, territory_embed_dim)
    """

    def __init__(self, territory_embed_dim: int = 64):
        super().__init__()
        # Raw per-territory features: owner(7) + units(7*13=91) + production(1) + water(1)
        #   + factory_damage(1) + conquered(1) + is_vc(1) = 103
        raw_per_terr = NUM_PLAYERS + NUM_PLAYERS * NUM_UNIT_TYPES + 5
        # We compress: owner(7) + friendly_units(13) + enemy_units(13) + friendly_strength(1)
        #   + enemy_strength(1) + production(1) + water(1) + factory_dmg(1) + conquered(1) + vc(1) = 40
        compressed_dim = 40

        self.compress = nn.Sequential(
            nn.Linear(raw_per_terr, compressed_dim),
            nn.ReLU(),
        )
        self.embed = nn.Sequential(
            nn.Linear(compressed_dim, territory_embed_dim),
            nn.LayerNorm(territory_embed_dim),
            nn.ReLU(),
        )

    def forward(self, obs, is_axis_player: torch.Tensor = None):
        batch = obs.shape[0]
        raw_per_terr = NUM_PLAYERS + NUM_PLAYERS * NUM_UNIT_TYPES + 5
        global_dim = NUM_PLAYERS * 2 + 1  # pus(7) + round(1) + player_onehot(7)

        # Split territory features from global
        terr_features = obs[:, :NUM_TERRITORIES * raw_per_terr].reshape(batch, NUM_TERRITORIES, raw_per_terr)
        global_features = obs[:, NUM_TERRITORIES * raw_per_terr:]

        # Compress and embed
        compressed = self.compress(terr_features)
        embedded = self.embed(compressed)  # (batch, 162, embed_dim)

        return embedded, global_features


class GNNLayer(nn.Module):
    """Sparse message-passing GNN layer.

    Instead of dense O(T²) attention, uses pre-computed normalized adjacency
    for neighbor aggregation. Single matmul per layer, no masking/softmax.
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.W_self = nn.Linear(embed_dim, embed_dim)
        self.W_neighbor = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.ff_norm = nn.LayerNorm(embed_dim)

    def forward(self, x, adj_norm):
        """
        x: (batch, num_territories, embed_dim)
        adj_norm: (T, T) float — row-normalized adjacency with self-loops
        """
        # Neighbor aggregation: adj_norm @ W_neighbor(x) — one sparse matmul
        neighbor_msg = torch.matmul(adj_norm, self.W_neighbor(x))  # (B, T, D)
        self_msg = self.W_self(x)  # (B, T, D)
        out = F.relu(self_msg + neighbor_msg)

        # Skip connection + norm
        x = self.norm(x + out)

        # Feed-forward with skip
        x = self.ff_norm(x + self.ff(x))

        return x


class GlobalAttentionPool(nn.Module):
    """Attention-weighted pooling over territories.

    Learns which territories are most relevant for the current decision,
    then produces a fixed-size global representation.
    """

    def __init__(self, embed_dim: int, num_heads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        # Learned query tokens for: purchase decision, attack decision, reinforce decision
        self.query_tokens = nn.Parameter(torch.randn(3, embed_dim) * 0.02)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """
        x: (batch, num_territories, embed_dim)
        returns: (batch, 3 * embed_dim) — pooled features for purchase/attack/reinforce
        """
        batch = x.shape[0]
        queries = self.query_tokens.unsqueeze(0).expand(batch, -1, -1)  # (B, 3, D)
        pooled, _ = self.attn(queries, x, x)  # (B, 3, D)
        pooled = self.norm(pooled)
        return pooled.reshape(batch, -1)  # (B, 3*D)


class ActorCriticV3(nn.Module):
    """Territory-aware architecture with sparse GNN + attention pooling.

    Architecture:
    1. TerritoryEncoder: compress raw obs into per-territory embeddings
    2. Sparse GNN layers: message passing along pre-computed adjacency (2 layers)
    3. GlobalAttentionPool: 3 learned queries attend to territories
    4. Policy heads: purchase (from pooled), attack/reinforce (per-territory)
    5. Value head: from pooled global representation
    """

    def __init__(
        self,
        obs_size: int,
        action_dim: int,
        territory_embed_dim: int = 64,
        num_gnn_layers: int = 2,
        num_heads: int = 2,
        adj_matrix: torch.Tensor = None,
    ):
        super().__init__()
        self.obs_size = obs_size
        self.action_dim = action_dim
        self.territory_embed_dim = territory_embed_dim
        self.num_territories = NUM_TERRITORIES
        self.num_unit_types = NUM_UNIT_TYPES

        # Pre-compute row-normalized adjacency with self-loops (float buffer)
        if adj_matrix is not None:
            adj = adj_matrix.float()
            adj = adj + torch.eye(NUM_TERRITORIES)  # add self-loops
            adj = adj / adj.sum(dim=1, keepdim=True).clamp(min=1)  # row-normalize
            self.register_buffer('adj_norm', adj)
        else:
            adj = torch.eye(NUM_TERRITORIES)
            self.register_buffer('adj_norm', adj)

        # 1. Territory encoder
        self.territory_encoder = TerritoryEncoder(territory_embed_dim)

        # 2. Global features encoder
        global_dim = NUM_PLAYERS * 2 + 1  # 15
        self.global_encoder = nn.Sequential(
            nn.Linear(global_dim, 64),
            nn.ReLU(),
            nn.Linear(64, territory_embed_dim),
            nn.ReLU(),
        )

        # 3. Sparse GNN layers
        self.gnn_layers = nn.ModuleList([
            GNNLayer(territory_embed_dim) for _ in range(num_gnn_layers)
        ])

        # 4. Global attention pooling
        self.global_pool = GlobalAttentionPool(territory_embed_dim, num_heads)

        # 5. Policy heads
        pooled_dim = 3 * territory_embed_dim + territory_embed_dim  # pool + global

        # Purchase head (from pooled global representation)
        self.purchase_head = nn.Sequential(
            nn.Linear(pooled_dim, 64),
            nn.ReLU(),
            nn.Linear(64, NUM_UNIT_TYPES),
            nn.Sigmoid(),
        )

        # Attack head (per-territory scores from territory embeddings)
        self.attack_head = nn.Sequential(
            nn.Linear(territory_embed_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        # Reinforce head (per-territory scores)
        self.reinforce_head = nn.Sequential(
            nn.Linear(territory_embed_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        # 6. Value head
        self.value_head = nn.Sequential(
            nn.Linear(pooled_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        # 7. Log std for exploration
        self.log_std = nn.Parameter(torch.zeros(action_dim) - 0.5)

    def forward(self, obs):
        batch = obs.shape[0]

        # Encode territories
        terr_embed, global_raw = self.territory_encoder(obs)  # (B, T, D), (B, 15)

        # Encode global features and broadcast
        global_embed = self.global_encoder(global_raw)  # (B, D)

        # Add global context to each territory embedding
        terr_embed = terr_embed + global_embed.unsqueeze(1)

        # Sparse GNN message passing
        for gnn in self.gnn_layers:
            terr_embed = gnn(terr_embed, self.adj_norm)

        # Global attention pooling
        pooled = self.global_pool(terr_embed)  # (B, 3*D)
        global_ctx = torch.cat([pooled, global_embed], dim=-1)  # (B, 3*D + D)

        # Policy: purchase
        purchase_scores = self.purchase_head(global_ctx)  # (B, 13)

        # Policy: attack scores per territory
        attack_scores = self.attack_head(terr_embed).squeeze(-1)  # (B, 162)

        # Policy: reinforce scores per territory
        reinforce_scores = self.reinforce_head(terr_embed).squeeze(-1)  # (B, 162)

        # Combine into action vector (same format as V2 for compatibility)
        action_mean = torch.cat([purchase_scores, attack_scores, reinforce_scores], dim=-1)  # (B, 337)

        # Value
        value = self.value_head(global_ctx)  # (B, 1)

        return action_mean, value

    def get_action(self, obs, deterministic=False):
        action_mean, value = self.forward(obs)
        std = torch.exp(self.log_std).expand_as(action_mean)
        if deterministic:
            action = action_mean
        else:
            dist = torch.distributions.Normal(action_mean, std)
            action = dist.sample().clamp(0.0, 1.0)
        dist = torch.distributions.Normal(action_mean, std)
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action.squeeze(0).detach().cpu().numpy(), log_prob.squeeze(0), value.squeeze(-1).squeeze(0)

    def evaluate_actions(self, obs, actions):
        action_mean, value = self.forward(obs)
        std = torch.exp(self.log_std).expand_as(action_mean)
        dist = torch.distributions.Normal(action_mean, std)
        return dist.log_prob(actions).sum(dim=-1), value.squeeze(-1), dist.entropy().sum(dim=-1)
