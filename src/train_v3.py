"""V3 PPO training with parallel vectorized environments.

Key improvements over V2:
- Multiprocessing vectorized environments (N processes in parallel)
- Batched inference (all envs forward pass at once)
- Better memory efficiency
- Progress tracking
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from .env_v2 import TripleAEnvV2
from .network_v2 import ActorCriticV2
from .vec_env import make_vec_env


@dataclass
class PPOConfigV3:
    num_envs: int = 16          # More envs = more parallelism
    max_rounds: int = 15
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coeff: float = 0.02
    value_coeff: float = 0.5
    max_grad_norm: float = 0.5
    num_steps: int = 128        # Steps per env per rollout
    num_epochs: int = 4
    batch_size: int = 256       # Larger batches
    total_timesteps: int = 2_000_000
    hidden_size: int = 512
    log_dir: str = "runs_v3"
    save_dir: str = "checkpoints_v3"
    save_interval: int = 50_000
    anneal_lr: bool = True


def compute_gae(rewards, values, dones, next_values, gamma, gae_lambda):
    """Vectorized GAE computation across all environments."""
    num_steps, num_envs = rewards.shape
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros(num_envs)

    for t in reversed(range(num_steps)):
        if t == num_steps - 1:
            next_val = next_values
        else:
            next_val = values[t + 1]
        delta = rewards[t] + gamma * next_val * (1 - dones[t]) - values[t]
        advantages[t] = last_gae = delta + gamma * gae_lambda * (1 - dones[t]) * last_gae

    returns = advantages + values
    return advantages, returns


def train_v3(config: PPOConfigV3 | None = None):
    if config is None:
        config = PPOConfigV3()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Device: {device}")
    print(f"Parallel envs: {config.num_envs}")

    save_path = Path(config.save_dir)
    save_path.mkdir(exist_ok=True)
    writer = SummaryWriter(config.log_dir)

    # Create vectorized environment
    print("Starting parallel environment workers...")
    vec_env = make_vec_env(TripleAEnvV2, config.num_envs, max_rounds=config.max_rounds)

    obs_size = vec_env.observation_space.shape[0]
    action_dim = vec_env.action_dim

    model = ActorCriticV2(obs_size, action_dim, config.hidden_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, eps=1e-5)

    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Obs: {obs_size}, Actions: {action_dim}")

    # Rollout storage
    obs_buf = torch.zeros((config.num_steps, config.num_envs, obs_size))
    act_buf = torch.zeros((config.num_steps, config.num_envs, action_dim))
    logp_buf = torch.zeros((config.num_steps, config.num_envs))
    rew_buf = torch.zeros((config.num_steps, config.num_envs))
    val_buf = torch.zeros((config.num_steps, config.num_envs))
    done_buf = torch.zeros((config.num_steps, config.num_envs))

    # Initialize
    current_obs, _ = vec_env.reset()

    global_step = 0
    episode_returns = []
    episode_wins = []
    episode_rounds = []
    best_win_rate = 0.0
    start_time = time.time()

    num_updates = config.total_timesteps // (config.num_steps * config.num_envs)
    batch_size = config.num_steps * config.num_envs

    for update in range(num_updates):
        if config.anneal_lr:
            frac = 1.0 - update / num_updates
            for pg in optimizer.param_groups:
                pg["lr"] = config.learning_rate * frac

        # === Collect rollout ===
        model.eval()
        for step in range(config.num_steps):
            obs_tensor = torch.FloatTensor(current_obs).to(device)

            with torch.no_grad():
                action_mean, values = model.forward(obs_tensor)
                std = torch.exp(model.log_std).expand_as(action_mean)
                dist = torch.distributions.Normal(action_mean, std)
                actions = dist.sample()
                actions = torch.clamp(actions, 0.0, 1.0)
                log_probs = dist.log_prob(actions).sum(dim=-1)

            obs_buf[step] = obs_tensor.cpu()
            act_buf[step] = actions.cpu()
            logp_buf[step] = log_probs.cpu()
            val_buf[step] = values.squeeze(-1).cpu()

            # Step all envs in parallel
            actions_np = actions.cpu().numpy()
            next_obs, rewards, terminateds, truncateds, infos = vec_env.step(actions_np)

            rew_buf[step] = torch.FloatTensor(rewards)
            done_buf[step] = torch.FloatTensor(terminateds | truncateds)

            # Track episodes
            for i, info in enumerate(infos):
                if info.get("_final_obs", False) or terminateds[i] or truncateds[i]:
                    ep_vc = info.get("allied_vc", 0) - info.get("axis_vc", 0)
                    episode_returns.append(ep_vc)
                    episode_wins.append(1.0 if info.get("winner") == "Allies" else 0.0)
                    episode_rounds.append(info.get("round", 0))

            current_obs = next_obs
            global_step += config.num_envs

        # === Compute advantages ===
        with torch.no_grad():
            next_obs_t = torch.FloatTensor(current_obs).to(device)
            _, next_vals = model.forward(next_obs_t)
            next_vals = next_vals.squeeze(-1).cpu()

        advantages, returns = compute_gae(
            rew_buf, val_buf, done_buf, next_vals,
            config.gamma, config.gae_lambda,
        )

        # Flatten
        obs_flat = obs_buf.reshape(-1, obs_size).to(device)
        act_flat = act_buf.reshape(-1, action_dim).to(device)
        logp_flat = logp_buf.reshape(-1).to(device)
        adv_flat = advantages.reshape(-1).to(device)
        ret_flat = returns.reshape(-1).to(device)

        adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)

        # === PPO update ===
        model.train()
        pg_losses = []
        v_losses = []
        ent_losses = []

        for epoch in range(config.num_epochs):
            indices = torch.randperm(batch_size)
            for start in range(0, batch_size, config.batch_size):
                end = min(start + config.batch_size, batch_size)
                mb = indices[start:end]

                new_logp, new_val, entropy = model.evaluate_actions(
                    obs_flat[mb], act_flat[mb]
                )

                ratio = torch.exp(new_logp - logp_flat[mb])
                surr1 = ratio * adv_flat[mb]
                surr2 = torch.clamp(ratio, 1 - config.clip_epsilon,
                                    1 + config.clip_epsilon) * adv_flat[mb]
                pg_loss = -torch.min(surr1, surr2).mean()
                v_loss = nn.MSELoss()(new_val, ret_flat[mb])
                ent_loss = -entropy.mean()

                loss = pg_loss + config.value_coeff * v_loss + config.entropy_coeff * ent_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()

                pg_losses.append(pg_loss.item())
                v_losses.append(v_loss.item())
                ent_losses.append(ent_loss.item())

        # === Logging ===
        if update % 5 == 0 and episode_returns:
            n = min(50, len(episode_returns))
            avg_vc = np.mean(episode_returns[-n:])
            win_rate = np.mean(episode_wins[-n:])
            avg_rnd = np.mean(episode_rounds[-n:]) if episode_rounds else 0
            elapsed = time.time() - start_time
            sps = global_step / max(elapsed, 1)

            writer.add_scalar("train/vc_diff", avg_vc, global_step)
            writer.add_scalar("train/win_rate", win_rate, global_step)
            writer.add_scalar("train/rounds", avg_rnd, global_step)
            writer.add_scalar("train/sps", sps, global_step)
            writer.add_scalar("loss/policy", np.mean(pg_losses), global_step)
            writer.add_scalar("loss/value", np.mean(v_losses), global_step)

            remaining_steps = config.total_timesteps - global_step
            eta_seconds = remaining_steps / max(sps, 1)
            eta_min = eta_seconds / 60
            if eta_min > 60:
                eta_str = f"{eta_min / 60:.1f}h"
            else:
                eta_str = f"{eta_min:.1f}m"
            pct = global_step / config.total_timesteps * 100

            print(f"U{update:4d}/{num_updates} | {global_step:8,}/{config.total_timesteps:,} ({pct:.0f}%) | "
                  f"{sps:5.0f} sps | ETA: {eta_str} | VC: {avg_vc:+.1f} | Win: {win_rate:.0%} | "
                  f"R{avg_rnd:.0f} | PG: {np.mean(pg_losses):.4f}")

            if win_rate > best_win_rate or (win_rate == best_win_rate and avg_vc > 0):
                best_win_rate = win_rate
                torch.save(model.state_dict(), save_path / "best_model.pt")

        if global_step % config.save_interval < config.num_steps * config.num_envs:
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": global_step,
            }, save_path / f"ckpt_{global_step}.pt")

    vec_env.close()
    torch.save(model.state_dict(), save_path / "final_model.pt")
    writer.close()

    total_time = time.time() - start_time
    print(f"\nDone! {global_step:,} steps in {total_time:.0f}s "
          f"({global_step / total_time:.0f} sps)")
    print(f"Best win rate: {best_win_rate:.0%}")
    print(f"Models: {save_path}")


if __name__ == "__main__":
    train_v3()
