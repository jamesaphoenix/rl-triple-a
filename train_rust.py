#!/usr/bin/env python3
"""PPO training with Rust game engine + PyTorch MPS GPU.

Rust engine: ~40,000 steps/sec (397x faster than Python)
Neural net: MPS GPU for forward/backward passes
No multiprocessing needed — Rust is fast enough single-threaded.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.game_data_export import export_map_arrays
from src.network_v2 import ActorCriticV2
from triplea_engine import TripleAEngine


def make_engine(seed=42):
    arrays = export_map_arrays()
    engine = TripleAEngine(
        arrays["adjacency"], arrays["is_water"], arrays["is_impassable"],
        arrays["production"], arrays["is_victory_city"], arrays["is_capital"],
        arrays["chinese_territories"],
        arrays["initial_units"], arrays["initial_owner"], arrays["initial_pus"],
        seed=seed,
    )
    # Register national objectives
    for no in arrays.get("national_objectives", []):
        engine.add_national_objective(
            no["player"], no["value"], no["territories"],
            no["count"], no["enemy_sea_zones"], no.get("allied_exclusion", False),
        )
    return engine


def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(T)):
        next_val = values[t + 1] if t + 1 < T else 0.0
        delta = rewards[t] + gamma * next_val * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages[t] = gae
    returns = advantages + np.array(values[:T])
    return advantages, returns


def train(
    num_envs=32,
    total_steps=2_000_000,
    num_steps=256,
    num_epochs=4,
    batch_size=512,
    lr=3e-4,
    save_dir="checkpoints_rust",
    max_rounds=15,
):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)

    # Create engines
    engines = [make_engine(seed=i) for i in range(num_envs)]
    num_t = engines[0].get_num_territories()
    obs_size = engines[0].get_obs_size()
    action_dim = 13 + num_t + num_t  # purchase + attack_scores + reinforce_scores

    print(f"Envs: {num_envs} | Obs: {obs_size} | Actions: {action_dim}")

    model = ActorCriticV2(obs_size, action_dim, hidden_size=512).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, eps=1e-5)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    # Initialize
    current_obs = np.zeros((num_envs, obs_size), dtype=np.float32)
    for i, eng in enumerate(engines):
        obs = eng.reset(i)
        current_obs[i] = np.array(obs)

    global_step = 0
    start_time = time.time()
    episode_rewards = []
    episode_wins = []
    episode_rounds = []
    best_win_rate = 0.0

    num_updates = total_steps // (num_steps * num_envs)

    for update in range(num_updates):
        # LR annealing
        frac = 1.0 - update / num_updates
        for pg in optimizer.param_groups:
            pg["lr"] = lr * frac

        # === Collect rollout (Rust engines — fast!) ===
        all_obs = np.zeros((num_steps, num_envs, obs_size), dtype=np.float32)
        all_actions = np.zeros((num_steps, num_envs, action_dim), dtype=np.float32)
        all_logprobs = np.zeros((num_steps, num_envs), dtype=np.float32)
        all_rewards = np.zeros((num_steps, num_envs), dtype=np.float32)
        all_values = np.zeros((num_steps, num_envs), dtype=np.float32)
        all_dones = np.zeros((num_steps, num_envs), dtype=np.float32)

        model.eval()
        for step in range(num_steps):
            obs_tensor = torch.from_numpy(current_obs).to(device)

            with torch.no_grad():
                action_mean, values = model.forward(obs_tensor)
                std = torch.exp(model.log_std).expand_as(action_mean)
                dist = torch.distributions.Normal(action_mean, std)
                actions = dist.sample().clamp(0.0, 1.0)
                log_probs = dist.log_prob(actions).sum(dim=-1)

            actions_np = actions.cpu().numpy()
            values_np = values.squeeze(-1).cpu().numpy()
            logprobs_np = log_probs.cpu().numpy()

            all_obs[step] = current_obs
            all_actions[step] = actions_np
            all_logprobs[step] = logprobs_np
            all_values[step] = values_np

            # Step all engines (single-threaded — still fast!)
            for i in range(num_envs):
                a = actions_np[i]
                purchase = a[:13].astype(np.float32)
                attack = a[13:13+num_t].astype(np.float32)
                reinforce = a[13+num_t:13+2*num_t].astype(np.float32)

                result = engines[i].step(purchase, attack, reinforce)
                all_rewards[step, i] = result["reward"]
                all_dones[step, i] = float(result["done"])
                current_obs[i] = np.array(result["obs"])

                if result["done"]:
                    episode_rewards.append(result["reward"])
                    episode_wins.append(1.0 if result["winner"] == 1 else 0.0)
                    episode_rounds.append(result["round"])
                    current_obs[i] = np.array(engines[i].reset(global_step + i))

            global_step += num_envs

        # === Compute GAE ===
        all_advantages = np.zeros_like(all_rewards)
        all_returns = np.zeros_like(all_rewards)
        for e in range(num_envs):
            values_list = list(all_values[:, e])
            adv, ret = compute_gae(all_rewards[:, e], values_list, all_dones[:, e])
            all_advantages[:, e] = adv
            all_returns[:, e] = ret

        # Flatten
        N = num_steps * num_envs
        obs_flat = torch.from_numpy(all_obs.reshape(N, -1)).to(device)
        act_flat = torch.from_numpy(all_actions.reshape(N, -1)).to(device)
        logp_flat = torch.from_numpy(all_logprobs.reshape(N)).to(device)
        adv_flat = torch.from_numpy(all_advantages.reshape(N)).to(device)
        ret_flat = torch.from_numpy(all_returns.reshape(N)).to(device)

        adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)

        # === PPO Update (on GPU) ===
        model.train()
        pg_losses = []
        v_losses = []

        for epoch in range(num_epochs):
            perm = torch.randperm(N, device=device)
            for start in range(0, N, batch_size):
                end = min(start + batch_size, N)
                idx = perm[start:end]

                new_logp, new_val, entropy = model.evaluate_actions(
                    obs_flat[idx], act_flat[idx]
                )

                ratio = torch.exp(new_logp - logp_flat[idx])
                surr1 = ratio * adv_flat[idx]
                surr2 = torch.clamp(ratio, 0.8, 1.2) * adv_flat[idx]
                pg_loss = -torch.min(surr1, surr2).mean()
                v_loss = nn.MSELoss()(new_val, ret_flat[idx])
                ent_loss = -entropy.mean()

                loss = pg_loss + 0.5 * v_loss + 0.02 * ent_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()

                pg_losses.append(pg_loss.item())
                v_losses.append(v_loss.item())

        # === Logging ===
        if update % 2 == 0:
            elapsed = time.time() - start_time
            sps = global_step / max(elapsed, 1)
            remaining = (total_steps - global_step) / max(sps, 1)
            eta = f"{remaining/60:.1f}m" if remaining < 3600 else f"{remaining/3600:.1f}h"
            pct = global_step / total_steps * 100

            n = min(50, len(episode_wins)) if episode_wins else 0
            win_rate = np.mean(episode_wins[-n:]) if n > 0 else 0
            avg_reward = np.mean(episode_rewards[-n:]) if n > 0 else 0
            avg_round = np.mean(episode_rounds[-n:]) if n > 0 else 0

            print(f"U{update:4d}/{num_updates} | {global_step:>9,}/{total_steps:,} ({pct:4.1f}%) | "
                  f"{sps:>7,.0f} sps | ETA: {eta:>6s} | "
                  f"Win: {win_rate:4.0%} | R: {avg_reward:>+6.1f} | Rnd: {avg_round:4.0f} | "
                  f"PG: {np.mean(pg_losses):.4f}")

            if n > 0 and win_rate >= best_win_rate:
                best_win_rate = win_rate
                torch.save(model.state_dict(), save_path / "best_model.pt")

        if global_step % 200_000 < num_steps * num_envs:
            torch.save({"model": model.state_dict(), "step": global_step},
                       save_path / f"ckpt_{global_step}.pt")

    # Final save
    torch.save(model.state_dict(), save_path / "final_model.pt")
    total_time = time.time() - start_time
    print(f"\nDone! {global_step:,} steps in {total_time:.0f}s ({global_step/total_time:,.0f} sps)")
    print(f"Best win rate: {best_win_rate:.0%}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--total-steps", type=int, default=2_000_000)
    parser.add_argument("--num-steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--save-dir", type=str, default="checkpoints_rust")
    args = parser.parse_args()

    print("=" * 70)
    print("  RL TripleA — Rust Engine + MPS GPU Training")
    print(f"  Engine: Rust (PyO3) — ~40,000 game steps/sec")
    print(f"  Neural net: PyTorch on {('MPS GPU' if torch.backends.mps.is_available() else 'CPU')}")
    print(f"  Envs: {args.num_envs} | Steps: {args.total_steps:,}")
    print("=" * 70)

    train(
        num_envs=args.num_envs,
        total_steps=args.total_steps,
        num_steps=args.num_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        save_dir=args.save_dir,
    )


if __name__ == "__main__":
    main()
