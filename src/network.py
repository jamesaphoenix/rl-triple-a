"""V2 Actor-Critic network for multi-phase RL.

Uses a continuous action space interpreted differently per phase:
- Shared backbone extracts features from game state
- Phase-conditioned policy head outputs continuous actions
- Value head estimates expected return
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ActorCriticV2(nn.Module):
    """Actor-Critic with continuous action space for multi-phase play."""

    def __init__(self, obs_size: int, action_dim: int, hidden_size: int = 512):
        super().__init__()
        self.action_dim = action_dim

        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
        )

        # Policy head: outputs mean of Gaussian for each action dimension
        self.policy_mean = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Sigmoid(),  # output in [0, 1]
        )

        # Log standard deviation (learnable parameter)
        self.log_std = nn.Parameter(torch.zeros(action_dim) - 0.5)

        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, obs: torch.Tensor):
        features = self.shared(obs)
        action_mean = self.policy_mean(features)
        value = self.value_head(features)
        return action_mean, value

    def get_action(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, torch.Tensor, torch.Tensor]:
        action_mean, value = self.forward(obs)
        std = torch.exp(self.log_std).expand_as(action_mean)

        if deterministic:
            action = action_mean
        else:
            dist = torch.distributions.Normal(action_mean, std)
            action = dist.sample()
            action = torch.clamp(action, 0.0, 1.0)

        # Log probability
        dist = torch.distributions.Normal(action_mean, std)
        log_prob = dist.log_prob(action).sum(dim=-1)

        return (
            action.squeeze(0).detach().cpu().numpy(),
            log_prob.squeeze(0),
            value.squeeze(-1).squeeze(0),
        )

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        action_mean, value = self.forward(obs)
        std = torch.exp(self.log_std).expand_as(action_mean)

        dist = torch.distributions.Normal(action_mean, std)
        log_probs = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)

        return log_probs, value.squeeze(-1), entropy
