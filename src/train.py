"""PPO training loop for the Allied RL agent."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from .env import TripleAEnv
from .network import ActorCritic
from .units import NUM_UNIT_TYPES


@dataclass
class PPOConfig:
    # Environment
    num_envs: int = 8
    max_rounds: int = 15

    # PPO hyperparameters
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coeff: float = 0.01
    value_coeff: float = 0.5
    max_grad_norm: float = 0.5

    # Training
    num_steps: int = 128  # steps per rollout
    num_epochs: int = 4  # PPO epochs per rollout
    batch_size: int = 64
    total_timesteps: int = 1_000_000

    # Network
    hidden_size: int = 512

    # Logging
    log_dir: str = "runs"
    save_dir: str = "checkpoints"
    save_interval: int = 10_000
    log_interval: int = 1_000


@dataclass
class RolloutBuffer:
    obs: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    log_probs: list = field(default_factory=list)
    rewards: list = field(default_factory=list)
    values: list = field(default_factory=list)
    dones: list = field(default_factory=list)
    budgets: list = field(default_factory=list)

    def clear(self):
        self.obs.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()
        self.budgets.clear()


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    next_value: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute Generalized Advantage Estimation."""
    num_steps = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    last_gae = 0.0

    for t in reversed(range(num_steps)):
        if t == num_steps - 1:
            next_val = next_value
        else:
            next_val = values[t + 1]

        delta = rewards[t] + gamma * next_val * (1 - dones[t]) - values[t]
        advantages[t] = last_gae = delta + gamma * gae_lambda * (1 - dones[t]) * last_gae

    returns = advantages + values
    return advantages, returns


def train(config: PPOConfig | None = None):
    """Main training loop."""
    if config is None:
        config = PPOConfig()

    # Use CPU for stability (MPS can cause NaN in softmax)
    # Switch to CUDA if available for speed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create directories
    save_path = Path(config.save_dir)
    save_path.mkdir(exist_ok=True)

    writer = SummaryWriter(config.log_dir)

    # Create environments
    envs = [TripleAEnv(seed=i, max_rounds=config.max_rounds) for i in range(config.num_envs)]

    # Get observation size
    obs_size = envs[0].state.observation_size

    # Create network
    model = ActorCritic(obs_size, config.hidden_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Observation size: {obs_size}")

    # Initialize environments
    current_obs = []
    current_budgets = []
    for env in envs:
        obs, info = env.reset()
        current_obs.append(obs)
        current_budgets.append(info.get("pus", 0))

    # Training loop
    global_step = 0
    episode_rewards = []
    episode_lengths = []
    episode_wins = []
    best_win_rate = 0.0

    num_updates = config.total_timesteps // (config.num_steps * config.num_envs)

    for update in range(num_updates):
        buffer = RolloutBuffer()

        # Collect rollout
        model.eval()
        for step in range(config.num_steps):
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(np.array(current_obs)).to(device)

                actions = []
                log_probs = []
                values = []

                for i in range(config.num_envs):
                    obs_i = obs_tensor[i:i+1]
                    action, log_prob, value = model.get_action(
                        obs_i, current_budgets[i]
                    )
                    actions.append(action)
                    log_probs.append(log_prob.item())
                    values.append(value.item())

            # Step environments
            next_obs = []
            next_budgets = []
            for i, env in enumerate(envs):
                obs, reward, terminated, truncated, info = env.step(actions[i])
                done = terminated or truncated

                buffer.obs.append(current_obs[i])
                buffer.actions.append(actions[i])
                buffer.log_probs.append(log_probs[i])
                buffer.rewards.append(reward)
                buffer.values.append(values[i])
                buffer.dones.append(float(done))
                buffer.budgets.append(current_budgets[i])

                if done:
                    episode_rewards.append(sum(buffer.rewards[-config.num_steps:]))
                    episode_lengths.append(info.get("round", 0))
                    episode_wins.append(1.0 if info.get("winner") == "Allies" else 0.0)
                    obs, info = env.reset()

                next_obs.append(obs)
                next_budgets.append(info.get("pus", 0))
                global_step += 1

            current_obs = next_obs
            current_budgets = next_budgets

        # Compute advantages
        model.eval()
        with torch.no_grad():
            next_obs_tensor = torch.FloatTensor(np.array(current_obs)).to(device)
            _, next_values = model.forward(next_obs_tensor)
            next_values = next_values.squeeze(-1).cpu()

        obs_batch = torch.FloatTensor(np.array(buffer.obs)).to(device)
        actions_batch = torch.LongTensor(np.array(buffer.actions)).to(device)
        old_log_probs = torch.FloatTensor(buffer.log_probs).to(device)
        rewards_batch = torch.FloatTensor(buffer.rewards)
        values_batch = torch.FloatTensor(buffer.values)
        dones_batch = torch.FloatTensor(buffer.dones)
        budgets_batch = torch.FloatTensor(buffer.budgets).to(device)

        # Reshape for GAE (num_steps, num_envs) -> flatten
        num_samples = len(buffer.rewards)

        advantages, returns = compute_gae(
            rewards_batch, values_batch, dones_batch,
            next_values.mean(),  # approximate
            config.gamma, config.gae_lambda,
        )
        advantages = advantages.to(device)
        returns = returns.to(device)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update
        model.train()
        for epoch in range(config.num_epochs):
            # Mini-batch updates
            indices = torch.randperm(num_samples)
            for start in range(0, num_samples, config.batch_size):
                end = min(start + config.batch_size, num_samples)
                mb_idx = indices[start:end]

                mb_obs = obs_batch[mb_idx]
                mb_actions = actions_batch[mb_idx]
                mb_old_log_probs = old_log_probs[mb_idx]
                mb_advantages = advantages[mb_idx]
                mb_returns = returns[mb_idx]
                mb_budgets = budgets_batch[mb_idx]

                new_log_probs, new_values, entropy = model.evaluate_action(
                    mb_obs, mb_actions, mb_budgets
                )

                # Policy loss (PPO clip)
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1 - config.clip_epsilon,
                                    1 + config.clip_epsilon) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = nn.MSELoss()(new_values, mb_returns)

                # Entropy bonus
                entropy_loss = -entropy.mean()

                # Total loss
                loss = (policy_loss +
                        config.value_coeff * value_loss +
                        config.entropy_coeff * entropy_loss)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()

        # Logging
        if update % 10 == 0 and episode_rewards:
            avg_reward = np.mean(episode_rewards[-100:])
            avg_length = np.mean(episode_lengths[-100:]) if episode_lengths else 0
            win_rate = np.mean(episode_wins[-100:]) if episode_wins else 0

            writer.add_scalar("train/avg_reward", avg_reward, global_step)
            writer.add_scalar("train/avg_episode_length", avg_length, global_step)
            writer.add_scalar("train/win_rate", win_rate, global_step)
            writer.add_scalar("train/policy_loss", policy_loss.item(), global_step)
            writer.add_scalar("train/value_loss", value_loss.item(), global_step)

            print(f"Update {update}/{num_updates} | Step {global_step} | "
                  f"Reward: {avg_reward:.1f} | WinRate: {win_rate:.1%} | "
                  f"Rounds: {avg_length:.1f}")

            # Save best model
            if win_rate > best_win_rate:
                best_win_rate = win_rate
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "win_rate": win_rate,
                    "global_step": global_step,
                }, save_path / "best_model.pt")

        # Periodic save
        if global_step % config.save_interval < config.num_steps * config.num_envs:
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "global_step": global_step,
            }, save_path / f"model_{global_step}.pt")

    # Final save
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "global_step": global_step,
    }, save_path / "final_model.pt")

    writer.close()
    print(f"\nTraining complete! Best win rate: {best_win_rate:.1%}")
    print(f"Models saved to {save_path}")


if __name__ == "__main__":
    train()
