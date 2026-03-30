"""Neural network for the Allied RL agent.

Architecture:
- Shared feature extractor (MLP for now, can upgrade to GNN later)
- Policy head: outputs purchase action probabilities
- Value head: estimates state value
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

from .units import NUM_UNIT_TYPES


class ActorCritic(nn.Module):
    """Actor-Critic network for PPO training."""

    def __init__(self, obs_size: int, hidden_size: int = 512):
        super().__init__()

        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
        )

        # Policy head: one output per unit type, representing "how many to buy"
        # We use a multi-head approach: each unit type gets its own small head
        # that outputs a distribution over 0-20 units
        self.policy_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Linear(64, 21),  # 0 to 20 units
            )
            for _ in range(NUM_UNIT_TYPES)
        ])

        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, obs: torch.Tensor):
        """Forward pass.

        Returns:
            action_logits: list of tensors, each (batch, 21) for each unit type
            value: (batch, 1) state value estimate
        """
        features = self.shared(obs)

        action_logits = [head(features) for head in self.policy_heads]
        value = self.value_head(features)

        return action_logits, value

    def get_action(
        self,
        obs: torch.Tensor,
        budget: int,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, torch.Tensor, torch.Tensor]:
        """Sample an action (purchase vector) respecting budget constraints.

        Args:
            obs: observation tensor (1, obs_size)
            budget: available PUs
            deterministic: if True, take argmax instead of sampling

        Returns:
            action: numpy array of shape (NUM_UNIT_TYPES,)
            log_prob: log probability of the action
            value: estimated state value
        """
        from .units import PURCHASABLE_UNITS

        action_logits, value = self.forward(obs)

        action = np.zeros(NUM_UNIT_TYPES, dtype=np.int32)
        total_log_prob = torch.tensor(0.0, device=obs.device)
        remaining_budget = budget

        # Sample each unit type sequentially, respecting budget
        for i, logits in enumerate(action_logits):
            ut = PURCHASABLE_UNITS[i]
            max_affordable = remaining_budget // ut.cost if ut.cost > 0 else 0
            max_affordable = min(max_affordable, 20)

            if max_affordable == 0:
                # Can't afford any - forced to buy 0
                action[i] = 0
                total_log_prob = total_log_prob + torch.tensor(0.0, device=obs.device)
                continue

            # Mask unaffordable options
            mask = torch.full((21,), float('-inf'), device=obs.device)
            mask[:max_affordable + 1] = 0.0

            masked_logits = logits.squeeze(0) + mask
            # Guard against NaN from MPS or gradient issues
            if torch.isnan(masked_logits).any():
                masked_logits = mask.clone()
                masked_logits[mask == 0.0] = 0.0
            probs = F.softmax(masked_logits, dim=-1)
            probs = torch.clamp(probs, min=1e-8)
            probs = probs / probs.sum()

            if deterministic:
                chosen = torch.argmax(probs)
            else:
                dist = Categorical(probs)
                chosen = dist.sample()

            count = chosen.item()
            action[i] = count
            remaining_budget -= count * ut.cost

            # Log prob
            log_p = F.log_softmax(masked_logits, dim=-1)
            total_log_prob = total_log_prob + log_p[chosen]

        return action, total_log_prob, value.squeeze(-1)

    def evaluate_action(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        budgets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate log probability and entropy for a batch of actions.

        Args:
            obs: (batch, obs_size)
            action: (batch, NUM_UNIT_TYPES) integer actions
            budgets: (batch,) PU budgets

        Returns:
            log_probs: (batch,)
            values: (batch,)
            entropy: (batch,)
        """
        from .units import PURCHASABLE_UNITS

        action_logits, values = self.forward(obs)
        batch_size = obs.shape[0]

        total_log_probs = torch.zeros(batch_size, device=obs.device)
        total_entropy = torch.zeros(batch_size, device=obs.device)

        for i, logits in enumerate(action_logits):
            ut = PURCHASABLE_UNITS[i]
            chosen = action[:, i].long()

            # Create budget mask for each sample in batch
            remaining = budgets.clone()
            for j in range(i):
                remaining -= action[:, j] * PURCHASABLE_UNITS[j].cost

            max_affordable = torch.clamp(remaining // max(ut.cost, 1), min=0, max=20)

            # Build mask - ensure at least option 0 is always valid
            mask = torch.full((batch_size, 21), float('-inf'), device=obs.device)
            for b in range(batch_size):
                num_valid = max(int(max_affordable[b].item()) + 1, 1)
                mask[b, :num_valid] = 0.0

            masked_logits = logits + mask
            log_probs = F.log_softmax(masked_logits, dim=-1)
            probs = F.softmax(masked_logits, dim=-1)

            # Clamp chosen to valid range to avoid index errors
            max_valid = max_affordable.long()
            chosen = torch.clamp(chosen, min=0, max=20)

            total_log_probs += log_probs.gather(1, chosen.unsqueeze(1)).squeeze(1)
            total_entropy += -(probs * log_probs).sum(dim=-1)

        return total_log_probs, values.squeeze(-1), total_entropy
