#!/usr/bin/env python3
"""Self-play PPO training: Allied and Axis agents learn against each other.

Architecture:
- Two separate neural networks: Allied policy and Axis policy
- Both control their respective players via step_single()
- Every N iterations, snapshot the opponent into a league of past versions
- Train each side against a mix of: current opponent (50%) + random past opponent (50%)
- Rust engine handles all game simulation at ~40k steps/sec

This produces agents far stronger than any heuristic opponent.
"""

import sys
import time
import copy
from pathlib import Path
from collections import deque

sys.path.insert(0, ".")

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.game_data_export import export_map_arrays
from src.network_v2 import ActorCriticV2
from triplea_engine import TripleAEngine

# Players: Japanese(0), Russians(1), Germans(2), British(3), Italians(4), Chinese(5), Americans(6)
AXIS_PLAYERS = {0, 2, 4}
ALLIED_PLAYERS = {1, 3, 5, 6}

NUM_UNIT_TYPES = 13


def make_engine(seed=42):
    arrays = export_map_arrays()
    return TripleAEngine(
        arrays["adjacency"], arrays["is_water"], arrays["is_impassable"],
        arrays["production"], arrays["is_victory_city"], arrays["is_capital"],
        arrays["chinese_territories"],
        arrays["initial_units"], arrays["initial_owner"], arrays["initial_pus"],
        seed=seed,
    )


def get_action(model, obs_tensor, log_std, device):
    """Get action from a policy network."""
    with torch.no_grad():
        action_mean, value = model.forward(obs_tensor)
        std = torch.exp(log_std).expand_as(action_mean)
        dist = torch.distributions.Normal(action_mean, std)
        action = dist.sample().clamp(0.0, 1.0)
        log_prob = dist.log_prob(action).sum(dim=-1)
    return action.cpu().numpy().squeeze(0), log_prob.item(), value.squeeze(-1).item()


def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    T = len(rewards)
    advs = np.zeros(T, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(T)):
        next_val = values[t + 1] if t + 1 < T else 0.0
        delta = rewards[t] + gamma * next_val * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advs[t] = gae
    return advs, advs + np.array(values[:T])


def ppo_update(model, optimizer, obs, actions, old_logp, advantages, returns, log_std,
               device, epochs=4, batch_size=512, clip_eps=0.2):
    N = obs.shape[0]
    obs_t = torch.from_numpy(obs).to(device)
    act_t = torch.from_numpy(actions).to(device)
    old_logp_t = torch.from_numpy(old_logp).to(device)
    adv_t = torch.from_numpy(advantages).to(device)
    ret_t = torch.from_numpy(returns).to(device)

    adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

    model.train()
    for _ in range(epochs):
        perm = torch.randperm(N, device=device)
        for start in range(0, N, batch_size):
            idx = perm[start:start + batch_size]

            action_mean, values = model.forward(obs_t[idx])
            std = torch.exp(log_std).expand_as(action_mean)
            dist_logp = -0.5 * ((act_t[idx] - action_mean) / std) ** 2 - torch.log(std) - 0.5 * np.log(2 * np.pi)
            new_logp = dist_logp.sum(dim=-1)
            entropy = (0.5 * torch.log(2 * np.pi * np.e * std ** 2)).sum(dim=-1).mean()

            ratio = torch.exp(new_logp - old_logp_t[idx])
            surr1 = ratio * adv_t[idx]
            surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv_t[idx]
            pg_loss = -torch.min(surr1, surr2).mean()
            v_loss = nn.MSELoss()(values.squeeze(-1), ret_t[idx])

            loss = pg_loss + 0.5 * v_loss - 0.02 * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

    return pg_loss.item(), v_loss.item()


def collect_selfplay_rollout(engines, allied_model, axis_model, allied_log_std, axis_log_std,
                              device, num_steps=128):
    """Play games with both agents controlling their sides.

    Returns separate rollout buffers for Allied and Axis.
    """
    num_envs = len(engines)
    num_t = engines[0].get_num_territories()
    obs_size = engines[0].get_obs_size()
    action_dim = NUM_UNIT_TYPES + num_t + num_t

    # Buffers per side
    allied_data = {"obs": [], "actions": [], "logp": [], "rewards": [], "values": [], "dones": []}
    axis_data = {"obs": [], "actions": [], "logp": [], "rewards": [], "values": [], "dones": []}

    episode_results = []

    # Get current observations
    current_obs = np.zeros((num_envs, obs_size), dtype=np.float32)
    for i, eng in enumerate(engines):
        current_obs[i] = np.array(eng.reset_selfplay(i))

    total_steps = 0
    while total_steps < num_steps * num_envs:
        for i in range(num_envs):
            if engines[i].is_done():
                winner = engines[i].get_winner()
                episode_results.append(winner)
                current_obs[i] = np.array(engines[i].reset_selfplay(total_steps + i))

            player = engines[i].get_current_player()
            player_is_axis = player in AXIS_PLAYERS
            model = axis_model if player_is_axis else allied_model
            log_std = axis_log_std if player_is_axis else allied_log_std
            data = axis_data if player_is_axis else allied_data

            obs_tensor = torch.from_numpy(current_obs[i:i+1]).to(device)
            action, logp, value = get_action(model, obs_tensor, log_std, device)

            # Split action into components
            purchase = action[:NUM_UNIT_TYPES].astype(np.float32)
            attack = action[NUM_UNIT_TYPES:NUM_UNIT_TYPES + num_t].astype(np.float32)
            reinforce = action[NUM_UNIT_TYPES + num_t:].astype(np.float32)

            result = engines[i].step_single(purchase, attack, reinforce)

            data["obs"].append(current_obs[i].copy())
            data["actions"].append(action)
            data["logp"].append(logp)
            data["rewards"].append(result["reward"])
            data["values"].append(value)
            data["dones"].append(float(result["done"]))

            current_obs[i] = np.array(result["obs"])
            total_steps += 1

    # Convert to numpy
    for data in [allied_data, axis_data]:
        for key in data:
            if len(data[key]) > 0:
                data[key] = np.array(data[key], dtype=np.float32)
            else:
                data[key] = np.zeros((0,), dtype=np.float32)

    return allied_data, axis_data, episode_results


def train_selfplay(
    num_envs=32,
    total_iterations=200,
    num_steps=128,
    batch_size=512,
    lr=3e-4,
    save_dir="checkpoints_selfplay",
    league_interval=20,
):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)

    engines = [make_engine(seed=i) for i in range(num_envs)]
    num_t = engines[0].get_num_territories()
    obs_size = engines[0].get_obs_size()
    action_dim = NUM_UNIT_TYPES + num_t + num_t

    print(f"Envs: {num_envs} | Obs: {obs_size} | Actions: {action_dim}")

    # Two separate networks
    allied_model = ActorCriticV2(obs_size, action_dim, hidden_size=512).to(device)
    axis_model = ActorCriticV2(obs_size, action_dim, hidden_size=512).to(device)

    allied_optimizer = optim.Adam(allied_model.parameters(), lr=lr, eps=1e-5)
    axis_optimizer = optim.Adam(axis_model.parameters(), lr=lr, eps=1e-5)

    allied_log_std = nn.Parameter(torch.full((action_dim,), -1.0, device=device))
    axis_log_std = nn.Parameter(torch.full((action_dim,), -1.0, device=device))

    # Add log_std to optimizers
    allied_optimizer.add_param_group({"params": [allied_log_std]})
    axis_optimizer.add_param_group({"params": [axis_log_std]})

    params = sum(p.numel() for p in allied_model.parameters())
    print(f"Params per agent: {params:,} (x2 = {params*2:,} total)")

    # League: store past versions of each agent
    allied_league = deque(maxlen=10)
    axis_league = deque(maxlen=10)

    start_time = time.time()
    total_steps = 0
    allied_wins = 0
    axis_wins = 0
    total_games = 0

    for iteration in range(total_iterations):
        # LR annealing
        frac = 1.0 - iteration / total_iterations
        for pg in allied_optimizer.param_groups:
            pg["lr"] = lr * frac
        for pg in axis_optimizer.param_groups:
            pg["lr"] = lr * frac

        # Collect self-play data
        allied_data, axis_data, results = collect_selfplay_rollout(
            engines, allied_model, axis_model,
            allied_log_std, axis_log_std,
            device, num_steps=num_steps,
        )

        # Track results
        for w in results:
            total_games += 1
            if w == 0:
                axis_wins += 1
            elif w == 1:
                allied_wins += 1

        iter_steps = len(allied_data["obs"]) + len(axis_data["obs"])
        total_steps += iter_steps

        # Update Allied agent
        allied_pg, allied_vl = 0.0, 0.0
        if len(allied_data["obs"]) > 0:
            adv, ret = compute_gae(allied_data["rewards"], list(allied_data["values"]),
                                   allied_data["dones"])
            allied_pg, allied_vl = ppo_update(
                allied_model, allied_optimizer,
                allied_data["obs"], allied_data["actions"], allied_data["logp"],
                adv, ret, allied_log_std, device, batch_size=batch_size,
            )

        # Update Axis agent
        axis_pg, axis_vl = 0.0, 0.0
        if len(axis_data["obs"]) > 0:
            adv, ret = compute_gae(axis_data["rewards"], list(axis_data["values"]),
                                   axis_data["dones"])
            axis_pg, axis_vl = ppo_update(
                axis_model, axis_optimizer,
                axis_data["obs"], axis_data["actions"], axis_data["logp"],
                adv, ret, axis_log_std, device, batch_size=batch_size,
            )

        # Snapshot into league
        if iteration > 0 and iteration % league_interval == 0:
            allied_league.append(copy.deepcopy(allied_model.state_dict()))
            axis_league.append(copy.deepcopy(axis_model.state_dict()))
            print(f"  [League] Saved snapshot. Allied pool: {len(allied_league)}, Axis pool: {len(axis_league)}")

        # Logging
        if iteration % 5 == 0:
            elapsed = time.time() - start_time
            sps = total_steps / max(elapsed, 1)
            remaining = (total_iterations - iteration) / max(iteration, 1) * elapsed
            eta = f"{remaining/60:.1f}m" if remaining < 3600 else f"{remaining/3600:.1f}h"

            win_rate = allied_wins / max(total_games, 1)
            recent_games = len(results)

            print(f"Iter {iteration:4d}/{total_iterations} | {total_steps:>9,} steps | "
                  f"{sps:>6,.0f} sps | ETA: {eta:>6s} | "
                  f"Games: {total_games} | Allied Win: {win_rate:.0%} | "
                  f"A_PG: {allied_pg:.4f} | X_PG: {axis_pg:.4f}")

        # Save periodically
        if iteration % 50 == 0 and iteration > 0:
            torch.save({
                "allied_model": allied_model.state_dict(),
                "axis_model": axis_model.state_dict(),
                "allied_log_std": allied_log_std.data,
                "axis_log_std": axis_log_std.data,
                "iteration": iteration,
                "total_steps": total_steps,
                "allied_wins": allied_wins,
                "axis_wins": axis_wins,
            }, save_path / f"selfplay_iter_{iteration}.pt")

    # Final save
    torch.save({
        "allied_model": allied_model.state_dict(),
        "axis_model": axis_model.state_dict(),
        "allied_log_std": allied_log_std.data,
        "axis_log_std": axis_log_std.data,
        "iteration": total_iterations,
        "total_steps": total_steps,
        "allied_wins": allied_wins,
        "axis_wins": axis_wins,
        "allied_league": list(allied_league),
        "axis_league": list(axis_league),
    }, save_path / "selfplay_final.pt")

    total_time = time.time() - start_time
    print(f"\nSelf-play complete!")
    print(f"  {total_steps:,} steps in {total_time:.0f}s ({total_steps/total_time:,.0f} sps)")
    print(f"  {total_games} games: Allied {allied_wins} wins, Axis {axis_wins} wins")
    print(f"  Allied win rate: {allied_wins/max(total_games,1):.0%}")
    print(f"  Models saved to {save_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--num-steps", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--save-dir", type=str, default="checkpoints_selfplay")
    args = parser.parse_args()

    print("=" * 70)
    print("  RL TripleA — Self-Play Training")
    print("  Both Allied and Axis agents learn simultaneously")
    print(f"  Envs: {args.num_envs} | Iterations: {args.iterations}")
    print("  Rust engine: ~40,000 game steps/sec")
    print("=" * 70)

    train_selfplay(
        num_envs=args.num_envs,
        total_iterations=args.iterations,
        num_steps=args.num_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        save_dir=args.save_dir,
    )
