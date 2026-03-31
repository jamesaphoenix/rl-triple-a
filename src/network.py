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
        # Raw per-territory features: owner(7) + units(7*13=91) + production(1) + water(1) = 100
        raw_per_terr = NUM_PLAYERS + NUM_PLAYERS * NUM_UNIT_TYPES + 2
        # We compress: owner(7) + friendly_units(13) + enemy_units(13) + friendly_strength(1)
        #   + enemy_strength(1) + production(1) + water(1) = 37
        compressed_dim = 37

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
        raw_per_terr = NUM_PLAYERS + NUM_PLAYERS * NUM_UNIT_TYPES + 2
        global_dim = NUM_PLAYERS * 2 + 1  # pus(7) + round(1) + player_onehot(7)

        # Split territory features from global
        terr_features = obs[:, :NUM_TERRITORIES * raw_per_terr].reshape(batch, NUM_TERRITORIES, raw_per_terr)
        global_features = obs[:, NUM_TERRITORIES * raw_per_terr:]

        # Compress and embed
        compressed = self.compress(terr_features)
        embedded = self.embed(compressed)  # (batch, 162, embed_dim)

        return embedded, global_features


class GNNLayer(nn.Module):
    """Graph neural network layer for territory message passing.

    Each territory aggregates information from its neighbors,
    weighted by learned attention scores.
    """

    def __init__(self, embed_dim: int, num_heads: int = 4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Message computation
        self.W_msg = nn.Linear(embed_dim, embed_dim)
        # Attention over neighbors
        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)
        # Output projection
        self.W_out = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        # Feed-forward
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.ff_norm = nn.LayerNorm(embed_dim)

    def forward(self, x, adj_mask):
        """
        x: (batch, num_territories, embed_dim)
        adj_mask: (num_territories, num_territories) bool — adjacency matrix
        """
        batch, T, D = x.shape
        H = self.num_heads
        Dh = self.head_dim

        # Multi-head attention restricted to adjacent territories
        q = self.W_q(x).reshape(batch, T, H, Dh).permute(0, 2, 1, 3)  # (B, H, T, Dh)
        k = self.W_k(x).reshape(batch, T, H, Dh).permute(0, 2, 1, 3)
        v = self.W_v(x).reshape(batch, T, H, Dh).permute(0, 2, 1, 3)

        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(Dh)  # (B, H, T, T)

        # Mask: only attend to adjacent territories (+ self)
        mask = adj_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, T, T)
        self_mask = torch.eye(T, device=x.device, dtype=torch.bool).unsqueeze(0).unsqueeze(0)
        full_mask = mask | self_mask  # adjacent + self
        scores = scores.masked_fill(~full_mask, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        attn = attn.masked_fill(torch.isnan(attn), 0.0)  # handle isolated nodes

        out = torch.matmul(attn, v)  # (B, H, T, Dh)
        out = out.permute(0, 2, 1, 3).reshape(batch, T, D)
        out = self.W_out(out)

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
    """Territory-aware architecture with GNN + attention.

    Architecture:
    1. TerritoryEncoder: compress raw obs into per-territory embeddings
    2. GNN layers: propagate information along adjacency graph (3 layers)
    3. GlobalAttentionPool: attend to relevant territories for each decision type
    4. Policy heads: purchase (from pooled), attack/reinforce (per-territory scores)
    5. Value head: from pooled global representation

    Total params: ~3-5M (vs 8.9M for V2) but much better capacity allocation.
    """

    def __init__(
        self,
        obs_size: int,
        action_dim: int,
        territory_embed_dim: int = 128,
        num_gnn_layers: int = 3,
        num_heads: int = 4,
        adj_matrix: torch.Tensor = None,
    ):
        super().__init__()
        self.obs_size = obs_size
        self.action_dim = action_dim
        self.territory_embed_dim = territory_embed_dim
        self.num_territories = NUM_TERRITORIES
        self.num_unit_types = NUM_UNIT_TYPES

        # Store adjacency matrix as buffer (not a parameter)
        if adj_matrix is not None:
            self.register_buffer('adj_mask', adj_matrix.bool())
        else:
            # Default: fully connected (will be overridden)
            self.register_buffer('adj_mask', torch.ones(NUM_TERRITORIES, NUM_TERRITORIES, dtype=torch.bool))

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

        # 3. GNN layers with skip connections
        self.gnn_layers = nn.ModuleList([
            GNNLayer(territory_embed_dim, num_heads) for _ in range(num_gnn_layers)
        ])

        # 4. Global attention pooling
        self.global_pool = GlobalAttentionPool(territory_embed_dim, num_heads)

        # 5. Policy heads
        pooled_dim = 3 * territory_embed_dim + territory_embed_dim  # pool + global

        # Purchase head (from pooled global representation)
        self.purchase_head = nn.Sequential(
            nn.Linear(pooled_dim, 128),
            nn.ReLU(),
            nn.Linear(128, NUM_UNIT_TYPES),
            nn.Sigmoid(),
        )

        # Attack head (per-territory scores from territory embeddings)
        self.attack_head = nn.Sequential(
            nn.Linear(territory_embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        # Reinforce head (per-territory scores)
        self.reinforce_head = nn.Sequential(
            nn.Linear(territory_embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        # 6. Value head
        self.value_head = nn.Sequential(
            nn.Linear(pooled_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
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

        # GNN message passing
        for gnn in self.gnn_layers:
            terr_embed = gnn(terr_embed, self.adj_mask)

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
