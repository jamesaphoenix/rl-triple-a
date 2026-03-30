#!/usr/bin/env python3
"""Self-play V2: batched inference, minimal Python↔Rust overhead.

Key speedup: group all envs by side (Allied/Axis), do ONE batched forward
pass per group, then step all engines. Eliminates 32 individual forward passes.
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

AXIS_PLAYERS = {0, 2, 4}
NUM_UNIT_TYPES = 13


def make_engine(seed=42):
    arrays = export_map_arrays()
    engine = TripleAEngine(
        arrays["adjacency"], arrays["is_water"], arrays["is_impassable"],
        arrays["production"], arrays["is_victory_city"], arrays["is_capital"],
        arrays["chinese_territories"],
        arrays["initial_units"], arrays["initial_owner"], arrays["initial_pus"],
        seed=seed,
    )
    for no in arrays.get("national_objectives", []):
        engine.add_national_objective(
            no["player"], no["value"], no["territories"],
            no["count"], no["enemy_sea_zones"], no.get("allied_exclusion", False),
        )
    return engine


def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    T = len(rewards)
    advs = np.zeros(T, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(T)):
        nv = values[t + 1] if t + 1 < T else 0.0
        delta = rewards[t] + gamma * nv * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advs[t] = gae
    return advs, advs + np.array(values[:T])


def ppo_update(model, optimizer, log_std, obs, actions, old_logp, advantages, returns,
               device, epochs=4, batch_size=512):
    N = obs.shape[0]
    if N == 0:
        return 0.0, 0.0
    obs_t = torch.from_numpy(obs).to(device)
    act_t = torch.from_numpy(actions).to(device)
    olp_t = torch.from_numpy(old_logp).to(device)
    adv_t = torch.from_numpy(advantages).to(device)
    ret_t = torch.from_numpy(returns).to(device)
    adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

    model.train()
    pg_l = v_l = 0.0
    for _ in range(epochs):
        perm = torch.randperm(N, device=device)
        for s in range(0, N, batch_size):
            idx = perm[s:s + batch_size]
            am, val = model.forward(obs_t[idx])
            std = torch.exp(log_std).expand_as(am)
            lp = (-0.5 * ((act_t[idx] - am) / std) ** 2 - torch.log(std) - 0.9189).sum(-1)
            ent = (0.5 * torch.log(2 * 3.14159 * 2.71828 * std ** 2)).sum(-1).mean()
            ratio = torch.exp(lp - olp_t[idx])
            pg = -torch.min(ratio * adv_t[idx],
                           torch.clamp(ratio, 0.8, 1.2) * adv_t[idx]).mean()
            vl = nn.MSELoss()(val.squeeze(-1), ret_t[idx])
            loss = pg + 0.5 * vl - 0.02 * ent
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            pg_l, v_l = pg.item(), vl.item()
    return pg_l, v_l


def train_selfplay(
    num_envs=32,
    total_iterations=300,
    steps_per_iter=4096,
    batch_size=512,
    lr=3e-4,
    save_dir="checkpoints_selfplay",
):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)

    engines = [make_engine(seed=i) for i in range(num_envs)]
    num_t = engines[0].get_num_territories()
    obs_size = engines[0].get_obs_size()
    action_dim = NUM_UNIT_TYPES + num_t + num_t

    print(f"Envs: {num_envs} | Obs: {obs_size} | Act: {action_dim}")

    allied_model = ActorCriticV2(obs_size, action_dim, hidden_size=512).to(device)
    axis_model = ActorCriticV2(obs_size, action_dim, hidden_size=512).to(device)
    allied_opt = optim.Adam(allied_model.parameters(), lr=lr, eps=1e-5)
    axis_opt = optim.Adam(axis_model.parameters(), lr=lr, eps=1e-5)
    allied_log_std = nn.Parameter(torch.full((action_dim,), -1.0, device=device))
    axis_log_std = nn.Parameter(torch.full((action_dim,), -1.0, device=device))
    allied_opt.add_param_group({"params": [allied_log_std]})
    axis_opt.add_param_group({"params": [axis_log_std]})

    params = sum(p.numel() for p in allied_model.parameters())
    print(f"Params per agent: {params:,}")

    league = deque(maxlen=10)

    # Initialize
    obs_arr = np.zeros((num_envs, obs_size), dtype=np.float32)
    for i in range(num_envs):
        obs_arr[i] = np.array(engines[i].reset_selfplay(i))

    start_time = time.time()
    total_steps = 0
    allied_wins = 0
    axis_wins = 0
    total_games = 0

    for iteration in range(total_iterations):
        frac = 1.0 - iteration / total_iterations
        for pg in allied_opt.param_groups:
            pg["lr"] = lr * frac
        for pg in axis_opt.param_groups:
            pg["lr"] = lr * frac

        # Collect data
        al_obs, al_act, al_lp, al_rew, al_val, al_done = [], [], [], [], [], []
        ax_obs, ax_act, ax_lp, ax_rew, ax_val, ax_done = [], [], [], [], [], []

        step_count = 0
        while step_count < steps_per_iter:
            # Group envs by current player side for batched inference
            axis_envs = []
            allied_envs = []
            for i in range(num_envs):
                if engines[i].is_done():
                    w = engines[i].get_winner()
                    total_games += 1
                    if w == 0: axis_wins += 1
                    elif w == 1: allied_wins += 1
                    obs_arr[i] = np.array(engines[i].reset_selfplay(total_steps + i))

                if engines[i].current_player_is_axis():
                    axis_envs.append(i)
                else:
                    allied_envs.append(i)

            # Batched forward pass for Allied envs
            if allied_envs:
                idxs = allied_envs
                obs_batch = torch.from_numpy(obs_arr[idxs]).to(device)
                with torch.no_grad():
                    am, val = allied_model.forward(obs_batch)
                    std = torch.exp(allied_log_std).expand_as(am)
                    dist = torch.distributions.Normal(am, std)
                    actions = dist.sample().clamp(0.0, 1.0)
                    logp = dist.log_prob(actions).sum(-1)
                actions_np = actions.cpu().numpy()
                logp_np = logp.cpu().numpy()
                val_np = val.squeeze(-1).cpu().numpy()

                for j, i in enumerate(idxs):
                    a = actions_np[j]
                    al_obs.append(obs_arr[i].copy())
                    al_act.append(a)
                    al_lp.append(logp_np[j])
                    al_val.append(val_np[j])

                    result = engines[i].step_single(
                        a[:NUM_UNIT_TYPES].astype(np.float32),
                        a[NUM_UNIT_TYPES:NUM_UNIT_TYPES+num_t].astype(np.float32),
                        a[NUM_UNIT_TYPES+num_t:].astype(np.float32),
                    )
                    al_rew.append(result["reward"])
                    al_done.append(float(result["done"]))
                    obs_arr[i] = np.array(result["obs"])
                    step_count += 1

            # Batched forward pass for Axis envs
            if axis_envs:
                idxs = axis_envs
                obs_batch = torch.from_numpy(obs_arr[idxs]).to(device)
                with torch.no_grad():
                    am, val = axis_model.forward(obs_batch)
                    std = torch.exp(axis_log_std).expand_as(am)
                    dist = torch.distributions.Normal(am, std)
                    actions = dist.sample().clamp(0.0, 1.0)
                    logp = dist.log_prob(actions).sum(-1)
                actions_np = actions.cpu().numpy()
                logp_np = logp.cpu().numpy()
                val_np = val.squeeze(-1).cpu().numpy()

                for j, i in enumerate(idxs):
                    a = actions_np[j]
                    ax_obs.append(obs_arr[i].copy())
                    ax_act.append(a)
                    ax_lp.append(logp_np[j])
                    ax_val.append(val_np[j])

                    result = engines[i].step_single(
                        a[:NUM_UNIT_TYPES].astype(np.float32),
                        a[NUM_UNIT_TYPES:NUM_UNIT_TYPES+num_t].astype(np.float32),
                        a[NUM_UNIT_TYPES+num_t:].astype(np.float32),
                    )
                    ax_rew.append(result["reward"])
                    ax_done.append(float(result["done"]))
                    obs_arr[i] = np.array(result["obs"])
                    step_count += 1

        total_steps += step_count

        # PPO updates
        def to_np(*lists):
            return tuple(np.array(l, dtype=np.float32) for l in lists)

        al_pg, al_vl = 0.0, 0.0
        if al_obs:
            o, a, l, r, v, d = to_np(al_obs, al_act, al_lp, al_rew, al_val, al_done)
            adv, ret = compute_gae(r, list(v), d)
            al_pg, al_vl = ppo_update(allied_model, allied_opt, allied_log_std,
                                       o, a, l, adv, ret, device, batch_size=batch_size)

        ax_pg, ax_vl = 0.0, 0.0
        if ax_obs:
            o, a, l, r, v, d = to_np(ax_obs, ax_act, ax_lp, ax_rew, ax_val, ax_done)
            adv, ret = compute_gae(r, list(v), d)
            ax_pg, ax_vl = ppo_update(axis_model, axis_opt, axis_log_std,
                                       o, a, l, adv, ret, device, batch_size=batch_size)

        # League snapshot
        if iteration > 0 and iteration % 30 == 0:
            league.append({
                "allied": copy.deepcopy(allied_model.state_dict()),
                "axis": copy.deepcopy(axis_model.state_dict()),
            })

        # Logging
        if iteration % 5 == 0:
            elapsed = time.time() - start_time
            sps = total_steps / max(elapsed, 1)
            rem = (total_iterations - iteration) / max(iteration, 1) * elapsed
            eta = f"{rem/60:.1f}m" if rem < 3600 else f"{rem/3600:.1f}h"
            wr = allied_wins / max(total_games, 1)

            print(f"I{iteration:4d}/{total_iterations} | {total_steps:>9,} steps | "
                  f"{sps:>6,.0f} sps | ETA: {eta:>6s} | "
                  f"Games: {total_games:>4d} | Allied: {wr:.0%} | "
                  f"A_pg: {al_pg:.4f} X_pg: {ax_pg:.4f}")

        # Save
        if iteration % 50 == 0 and iteration > 0:
            torch.save({
                "allied": allied_model.state_dict(),
                "axis": axis_model.state_dict(),
                "iteration": iteration,
                "allied_wins": allied_wins,
                "axis_wins": axis_wins,
                "total_games": total_games,
            }, save_path / f"selfplay_{iteration}.pt")

    # Final
    torch.save({
        "allied": allied_model.state_dict(),
        "axis": axis_model.state_dict(),
        "allied_log_std": allied_log_std.data,
        "axis_log_std": axis_log_std.data,
        "allied_wins": allied_wins,
        "axis_wins": axis_wins,
        "total_games": total_games,
        "league": list(league),
    }, save_path / "selfplay_final.pt")

    t = time.time() - start_time
    print(f"\nDone! {total_steps:,} steps in {t:.0f}s ({total_steps/t:,.0f} sps)")
    print(f"Games: {total_games} | Allied {allied_wins} wins ({allied_wins/max(total_games,1):.0%})")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--num-envs", type=int, default=32)
    p.add_argument("--iterations", type=int, default=300)
    p.add_argument("--steps-per-iter", type=int, default=4096)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--save-dir", type=str, default="checkpoints_selfplay")
    a = p.parse_args()

    print("=" * 70)
    print("  RL TripleA — Self-Play V2 (Batched Inference)")
    print(f"  Envs: {a.num_envs} | Iters: {a.iterations} | Steps/iter: {a.steps_per_iter}")
    print("  Rust engine + MPS GPU + batched forward passes")
    print("=" * 70)

    train_selfplay(
        num_envs=a.num_envs,
        total_iterations=a.iterations,
        steps_per_iter=a.steps_per_iter,
        batch_size=a.batch_size,
        lr=a.lr,
        save_dir=a.save_dir,
    )
