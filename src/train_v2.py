"""V2 PPO training with multi-phase environment and ProAI opponent."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from .env_v2 import TripleAEnvV2
from .network_v2 import ActorCriticV2


@dataclass
class PPOConfigV2:
    num_envs: int = 8
    max_rounds: int = 15
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coeff: float = 0.02
    value_coeff: float = 0.5
    max_grad_norm: float = 0.5
    num_steps: int = 256
    num_epochs: int = 4
    batch_size: int = 128
    total_timesteps: int = 2_000_000
    hidden_size: int = 512
    log_dir: str = "runs_v2"
    save_dir: str = "checkpoints_v2"
    save_interval: int = 50_000
    anneal_lr: bool = True


def compute_gae(rewards, values, dones, next_value, gamma, gae_lambda):
    num_steps = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    last_gae = 0.0
    for t in reversed(range(num_steps)):
        next_val = next_value if t == num_steps - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_val * (1 - dones[t]) - values[t]
        advantages[t] = last_gae = delta + gamma * gae_lambda * (1 - dones[t]) * last_gae
    returns = advantages + values
    return advantages, returns


def train_v2(config: PPOConfigV2 | None = None):
    if config is None:
        config = PPOConfigV2()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    save_path = Path(config.save_dir)
    save_path.mkdir(exist_ok=True)
    writer = SummaryWriter(config.log_dir)

    envs = [TripleAEnvV2(seed=i, max_rounds=config.max_rounds) for i in range(config.num_envs)]

    obs_size = envs[0].observation_space.shape[0]
    action_dim = envs[0].action_dim

    model = ActorCriticV2(obs_size, action_dim, config.hidden_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, eps=1e-5)

    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Obs size: {obs_size}, Action dim: {action_dim}")

    # Initialize
    current_obs = []
    for env in envs:
        obs, info = env.reset()
        current_obs.append(obs)

    global_step = 0
    episode_rewards = []
    episode_wins = []
    episode_rounds = []
    best_win_rate = 0.0

    num_updates = config.total_timesteps // (config.num_steps * config.num_envs)

    for update in range(num_updates):
        # Learning rate annealing
        if config.anneal_lr:
            frac = 1.0 - update / num_updates
            lr = config.learning_rate * frac
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

        # Collect rollout
        all_obs = []
        all_actions = []
        all_log_probs = []
        all_rewards = []
        all_values = []
        all_dones = []

        model.eval()
        for step in range(config.num_steps):
            obs_tensor = torch.FloatTensor(np.array(current_obs)).to(device)

            with torch.no_grad():
                actions = []
                log_probs = []
                values = []
                for i in range(config.num_envs):
                    obs_i = obs_tensor[i:i+1]
                    action, lp, val = model.get_action(obs_i)
                    actions.append(action)
                    log_probs.append(lp.item())
                    values.append(val.item())

            next_obs = []
            for i, env in enumerate(envs):
                obs, reward, terminated, truncated, info = env.step(actions[i])
                done = terminated or truncated

                all_obs.append(current_obs[i])
                all_actions.append(actions[i])
                all_log_probs.append(log_probs[i])
                all_rewards.append(reward)
                all_values.append(values[i])
                all_dones.append(float(done))

                if done:
                    ep_reward = info.get("allied_vc", 0) - info.get("axis_vc", 0)
                    episode_rewards.append(ep_reward)
                    episode_wins.append(1.0 if info.get("winner") == "Allies" else 0.0)
                    episode_rounds.append(info.get("round", 0))
                    obs, info = env.reset()

                next_obs.append(obs)
                global_step += 1

            current_obs = next_obs

        # Compute GAE
        model.eval()
        with torch.no_grad():
            next_obs_t = torch.FloatTensor(np.array(current_obs)).to(device)
            _, next_vals = model.forward(next_obs_t)
            next_vals = next_vals.squeeze(-1).mean().cpu()

        obs_batch = torch.FloatTensor(np.array(all_obs)).to(device)
        actions_batch = torch.FloatTensor(np.array(all_actions)).to(device)
        old_log_probs = torch.FloatTensor(all_log_probs).to(device)
        rewards_batch = torch.FloatTensor(all_rewards)
        values_batch = torch.FloatTensor(all_values)
        dones_batch = torch.FloatTensor(all_dones)

        advantages, returns = compute_gae(
            rewards_batch, values_batch, dones_batch,
            next_vals, config.gamma, config.gae_lambda,
        )
        advantages = advantages.to(device)
        returns = returns.to(device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update
        model.train()
        num_samples = len(all_rewards)
        pg_losses = []
        v_losses = []

        for epoch in range(config.num_epochs):
            indices = torch.randperm(num_samples)
            for start in range(0, num_samples, config.batch_size):
                end = min(start + config.batch_size, num_samples)
                mb_idx = indices[start:end]

                new_log_probs, new_values, entropy = model.evaluate_actions(
                    obs_batch[mb_idx], actions_batch[mb_idx]
                )

                ratio = torch.exp(new_log_probs - old_log_probs[mb_idx])
                surr1 = ratio * advantages[mb_idx]
                surr2 = torch.clamp(ratio, 1 - config.clip_epsilon,
                                    1 + config.clip_epsilon) * advantages[mb_idx]
                pg_loss = -torch.min(surr1, surr2).mean()
                v_loss = nn.MSELoss()(new_values, returns[mb_idx])
                entropy_loss = -entropy.mean()

                loss = pg_loss + config.value_coeff * v_loss + config.entropy_coeff * entropy_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()

                pg_losses.append(pg_loss.item())
                v_losses.append(v_loss.item())

        # Logging
        if update % 5 == 0 and episode_rewards:
            recent = min(50, len(episode_rewards))
            avg_reward = np.mean(episode_rewards[-recent:])
            win_rate = np.mean(episode_wins[-recent:])
            avg_rounds = np.mean(episode_rounds[-recent:]) if episode_rounds else 0

            writer.add_scalar("train/vc_diff", avg_reward, global_step)
            writer.add_scalar("train/win_rate", win_rate, global_step)
            writer.add_scalar("train/avg_rounds", avg_rounds, global_step)
            writer.add_scalar("train/pg_loss", np.mean(pg_losses), global_step)
            writer.add_scalar("train/value_loss", np.mean(v_losses), global_step)
            writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)

            print(f"Update {update}/{num_updates} | Step {global_step:,} | "
                  f"VCdiff: {avg_reward:+.1f} | Win: {win_rate:.0%} | "
                  f"Rounds: {avg_rounds:.0f} | PG: {np.mean(pg_losses):.4f}")

            if win_rate > best_win_rate:
                best_win_rate = win_rate
                torch.save(model.state_dict(), save_path / "best_model.pt")

        if global_step % config.save_interval < config.num_steps * config.num_envs:
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": global_step,
                "win_rate": best_win_rate,
            }, save_path / f"checkpoint_{global_step}.pt")

    torch.save(model.state_dict(), save_path / "final_model.pt")
    writer.close()
    print(f"\nDone! Best win rate: {best_win_rate:.0%}")
    print(f"Models saved to {save_path}")


if __name__ == "__main__":
    train_v2()
