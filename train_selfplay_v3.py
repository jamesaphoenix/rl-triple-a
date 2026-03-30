#!/usr/bin/env python3
"""Self-play V3: Rayon-parallel Rust batch engine + MPS GPU.

All 16 CPU cores run game simulations in parallel via Rayon.
GPU handles all neural net inference in single batched forward passes.
Zero Python-level loops over environments.
"""

import sys
import time
import copy
from pathlib import Path
from collections import deque

sys.path.insert(0, ".")
import os
import json
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.game_data_export import export_map_arrays
from src.network_v2 import ActorCriticV2
from triplea_engine import BatchEngine

NUM_UNIT_TYPES = 13


def compute_gae_batch(rewards, values, dones, last_values, gamma=0.99, lam=0.95):
    """Vectorized GAE across all envs.
    FIX #8: accepts last_values for bootstrap at rollout boundary.
    """
    T, N = rewards.shape
    advantages = np.zeros_like(rewards)
    gae = np.zeros(N, dtype=np.float32)
    for t in reversed(range(T)):
        # FIX #8: use bootstrap values for final step instead of zeros
        nv = values[t + 1] if t + 1 < T else last_values
        delta = rewards[t] + gamma * nv * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages[t] = gae
    return advantages, advantages + values


def train_selfplay(
    num_envs=128,
    total_iterations=500,
    steps_per_iter=256,
    num_epochs=4,
    batch_size=1024,
    lr=3e-4,
    save_dir="checkpoints_selfplay_v3",
):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    # File-based logging — bypasses conda stdout buffering
    log_dir = Path(save_dir)
    log_dir.mkdir(exist_ok=True)
    timing_log = open(log_dir / "timing.jsonl", "w")
    training_log = open(log_dir / "training.log", "w", buffering=1)  # line-buffered

    def log(msg):
        training_log.write(msg + "\n")
        print(msg, flush=True)

    def log_timing(data):
        timing_log.write(json.dumps(data) + "\n")
        timing_log.flush()

    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)

    # Create batch engine — all envs in one Rust object, parallel via Rayon
    arrays = export_map_arrays()
    batch_eng = BatchEngine(
        num_envs,
        arrays["adjacency"], arrays["is_water"], arrays["is_impassable"],
        arrays["production"], arrays["is_victory_city"], arrays["is_capital"],
        arrays["chinese_territories"],
        arrays["initial_units"], arrays["initial_owner"], arrays["initial_pus"],
    )
    # Register national objectives on all engines
    for no in arrays.get("national_objectives", []):
        batch_eng.add_national_objective(
            no["player"], no["value"], no["territories"],
            no["count"], no["enemy_sea_zones"], no.get("allied_exclusion", False),
        )

    num_t = batch_eng.get_num_territories()
    obs_size = batch_eng.get_obs_size()
    action_dim = NUM_UNIT_TYPES + num_t + num_t  # purchase + attack + reinforce

    print(f"Envs: {num_envs} | Obs: {obs_size} | Act: {action_dim}")
    print(f"Steps/iter: {steps_per_iter} | Batch: {batch_size}")
    print(f"Rayon parallel across all CPU cores")

    # Two networks
    # FIX #9: use model's internal log_std, don't create external parameter
    allied_model = ActorCriticV2(obs_size, action_dim, hidden_size=512).to(device)
    axis_model = ActorCriticV2(obs_size, action_dim, hidden_size=512).to(device)
    # FIX #10: include ALL parameters (including model.log_std) in optimizer
    allied_opt = optim.Adam(allied_model.parameters(), lr=lr, eps=1e-5)
    axis_opt = optim.Adam(axis_model.parameters(), lr=lr, eps=1e-5)
    # Use model's internal log_std instead of external parameter
    allied_log_std = allied_model.log_std
    axis_log_std = axis_model.log_std

    params = sum(p.numel() for p in allied_model.parameters())
    print(f"Params/agent: {params:,}")

    league = deque(maxlen=20)  # Store up to 20 past Axis snapshots
    league_model = ActorCriticV2(obs_size, action_dim, hidden_size=512).to(device)
    league_model.eval()

    # ── Resume from checkpoint or start fresh ──
    start_iteration = 0
    total_steps = 0
    allied_wins = 0
    axis_wins = 0
    total_games = 0

    # Find latest resumable checkpoint
    resume_path = None
    if os.path.exists(save_dir):
        import glob
        ckpts = sorted(glob.glob(str(save_path / "selfplay_*.pt")),
                       key=lambda f: os.path.getmtime(f))
        if ckpts:
            resume_path = ckpts[-1]

    if resume_path:
        log(f"  [Resume] Loading checkpoint: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        allied_model.load_state_dict(ckpt["allied"])
        axis_model.load_state_dict(ckpt["axis"])
        if "allied_opt" in ckpt:
            allied_opt.load_state_dict(ckpt["allied_opt"])
        if "axis_opt" in ckpt:
            axis_opt.load_state_dict(ckpt["axis_opt"])
        start_iteration = ckpt.get("iteration", 0)
        allied_wins = ckpt.get("allied_wins", 0)
        axis_wins = ckpt.get("axis_wins", 0)
        total_games = ckpt.get("total_games", 0)
        total_steps = ckpt.get("total_steps", 0)
        # Restore league
        for snap in ckpt.get("league", []):
            league.append(snap)
        log(f"  [Resume] Iter {start_iteration}, {total_games:,} games, "
            f"Allied {allied_wins/max(total_games,1):.0%}, league: {len(league)} snapshots")
    else:
        log(f"  [Fresh] No checkpoint found, starting from scratch")
        # Pre-seed league with foundation model's Axis weights if available
        foundation_path = "checkpoints_selfplay_v3/foundation_v1_282k_games_93pct.pt"
        if os.path.exists(foundation_path):
            try:
                fnd = torch.load(foundation_path, map_location="cpu", weights_only=False)
                if "axis" in fnd:
                    league.append(fnd["axis"])
                    log(f"  [League] Pre-seeded with foundation Axis (282k games)")
                for snap in fnd.get("league", []):
                    if "axis" in snap:
                        league.append(snap["axis"])
                if len(league) > 1:
                    log(f"  [League] Total pre-seeded snapshots: {len(league)}")
            except Exception as e:
                log(f"  [League] Could not load foundation for seeding: {e}")

    use_league = len(league) >= 2
    allied_log_std = allied_model.log_std
    axis_log_std = axis_model.log_std

    # Reset all engines
    all_obs_flat = np.array(batch_eng.reset_all())
    current_obs = all_obs_flat.reshape(num_envs, obs_size)

    start_time = time.time()

    for iteration in range(start_iteration, total_iterations):
        frac = 1.0 - iteration / total_iterations
        for pg in allied_opt.param_groups: pg["lr"] = lr * frac
        for pg in axis_opt.param_groups: pg["lr"] = lr * frac

        # Rollout storage
        all_obs_buf = np.zeros((steps_per_iter, num_envs, obs_size), dtype=np.float32)
        all_act_buf = np.zeros((steps_per_iter, num_envs, action_dim), dtype=np.float32)
        all_logp_buf = np.zeros((steps_per_iter, num_envs), dtype=np.float32)
        all_val_buf = np.zeros((steps_per_iter, num_envs), dtype=np.float32)
        all_rew_buf = np.zeros((steps_per_iter, num_envs), dtype=np.float32)
        all_done_buf = np.zeros((steps_per_iter, num_envs), dtype=np.float32)
        all_is_axis_buf = np.zeros((steps_per_iter, num_envs), dtype=np.float32)

        allied_model.eval()
        axis_model.eval()

        # Timing instrumentation
        t_gpu = 0.0
        t_rust = 0.0
        t_numpy = 0.0

        for step in range(steps_per_iter):
            t0 = time.time()

            is_axis_flags = np.array(batch_eng.get_is_axis())
            axis_mask = is_axis_flags.astype(bool)

            obs_tensor = torch.from_numpy(current_obs).to(device)

            t1 = time.time()
            # Batched GPU forward passes
            with torch.no_grad():
                # Allied always uses the main allied model
                al_mean, al_val = allied_model.forward(obs_tensor)
                al_std = torch.exp(allied_log_std).expand_as(al_mean)
                al_dist = torch.distributions.Normal(al_mean, al_std)
                al_actions = al_dist.sample().clamp(0.0, 1.0)
                al_logp = al_dist.log_prob(al_actions).sum(-1)

                # Axis: current model for all (used for training gradients)
                ax_mean, ax_val = axis_model.forward(obs_tensor)
                ax_std = torch.exp(axis_log_std).expand_as(ax_mean)
                ax_dist = torch.distributions.Normal(ax_mean, ax_std)
                ax_actions = ax_dist.sample().clamp(0.0, 1.0)
                ax_logp = ax_dist.log_prob(ax_actions).sum(-1)

                # LEAGUE: override some Axis envs with league opponents
                if use_league and len(league) >= 2:
                    league_ax_mean, _ = league_model.forward(obs_tensor)
                    league_ax_std = torch.exp(league_model.log_std).expand_as(league_ax_mean)
                    league_ax_dist = torch.distributions.Normal(league_ax_mean, league_ax_std)
                    league_ax_actions = league_ax_dist.sample().clamp(0.0, 1.0)

                    # Random noise actions (20% of Axis envs)
                    random_actions = torch.rand_like(ax_actions)

                    # Per-env opponent assignment:
                    # 40% current Axis, 40% league snapshot, 20% random
                    for i in range(num_envs):
                        if not axis_mask[i]:
                            continue
                        r = np.random.random()
                        if r < 0.40:
                            pass  # keep current axis
                        elif r < 0.80:
                            ax_actions[i] = league_ax_actions[i]  # league opponent
                        else:
                            ax_actions[i] = random_actions[i]  # random noise

            if device.type == 'mps':
                torch.mps.synchronize()
            t2 = time.time()
            t_gpu += t2 - t1

            actions_np = np.where(axis_mask[:, None],
                ax_actions.cpu().numpy(), al_actions.cpu().numpy())
            logp_np = np.where(axis_mask, ax_logp.cpu().numpy(), al_logp.cpu().numpy())
            val_np = np.where(axis_mask, ax_val.squeeze(-1).cpu().numpy(),
                             al_val.squeeze(-1).cpu().numpy())

            purchases = actions_np[:, :NUM_UNIT_TYPES].astype(np.float32)
            attack_scores = actions_np[:, NUM_UNIT_TYPES:NUM_UNIT_TYPES + num_t].astype(np.float32)
            reinforce_scores = actions_np[:, NUM_UNIT_TYPES + num_t:NUM_UNIT_TYPES + 2 * num_t].astype(np.float32)
            if reinforce_scores.shape[1] < num_t:
                pad = np.zeros((num_envs, num_t - reinforce_scores.shape[1]), dtype=np.float32)
                reinforce_scores = np.concatenate([reinforce_scores, pad], axis=1)

            t3 = time.time()
            t_numpy += t3 - t2

            # Rayon-parallel game step
            result = batch_eng.step_all(purchases, attack_scores, reinforce_scores)

            t4 = time.time()
            t_rust += t4 - t3

            rewards = np.array(result["rewards"])
            dones = np.array(result["dones"])
            winners = np.array(result["winners"])
            new_obs = np.array(result["obs"]).reshape(num_envs, obs_size)

            # Track episodes
            for i in range(num_envs):
                if dones[i] > 0.5:
                    total_games += 1
                    if winners[i] == 0: axis_wins += 1
                    elif winners[i] == 1: allied_wins += 1

            # Store
            all_obs_buf[step] = current_obs
            all_act_buf[step] = actions_np
            all_logp_buf[step] = logp_np
            all_val_buf[step] = val_np
            all_rew_buf[step] = rewards
            all_done_buf[step] = dones
            all_is_axis_buf[step] = is_axis_flags.astype(np.float32)

            current_obs = new_obs
            total_steps += num_envs

        # === Compute GAE ===
        # FIX #8: bootstrap value from final observation
        with torch.no_grad():
            obs_t = torch.from_numpy(current_obs).to(device)
            is_axis_now = np.array(batch_eng.get_is_axis()).astype(bool)
            _, al_boot = allied_model.forward(obs_t)
            _, ax_boot = axis_model.forward(obs_t)
            boot_vals = np.where(is_axis_now,
                                 ax_boot.squeeze(-1).cpu().numpy(),
                                 al_boot.squeeze(-1).cpu().numpy())
        advantages, returns = compute_gae_batch(all_rew_buf, all_val_buf, all_done_buf, boot_vals)

        # Split data by side
        al_mask = all_is_axis_buf < 0.5  # (steps, envs) bool
        ax_mask = ~al_mask

        def extract_side(mask):
            m = mask.reshape(-1)
            return (
                all_obs_buf.reshape(-1, obs_size)[m],
                all_act_buf.reshape(-1, action_dim)[m],
                all_logp_buf.reshape(-1)[m],
                advantages.reshape(-1)[m],
                returns.reshape(-1)[m],
            )

        al_o, al_a, al_l, al_adv, al_ret = extract_side(al_mask)
        ax_o, ax_a, ax_l, ax_adv, ax_ret = extract_side(ax_mask)

        # === PPO Updates ===
        def ppo_update(model, opt, log_std, obs, acts, old_lp, advs, rets, epochs_override=None, ent_coeff=0.02):
            N = obs.shape[0]
            if N == 0: return 0.0
            n_epochs = epochs_override or num_epochs
            o = torch.from_numpy(obs).to(device)
            a = torch.from_numpy(acts).to(device)
            ol = torch.from_numpy(old_lp).to(device)
            ad = torch.from_numpy(advs).to(device)
            ad = (ad - ad.mean()) / (ad.std() + 1e-8)
            rt = torch.from_numpy(rets).to(device)

            model.train()
            loss_val = 0.0
            for _ in range(n_epochs):
                perm = torch.randperm(N, device=device)
                for s in range(0, N, batch_size):
                    idx = perm[s:s+batch_size]
                    am, v = model.forward(o[idx])
                    std = torch.exp(log_std).expand_as(am)
                    lp = (-0.5 * ((a[idx] - am) / std)**2 - torch.log(std) - 0.9189).sum(-1)
                    ent = (0.5 * torch.log(2*3.14159*2.71828 * std**2)).sum(-1).mean()
                    ratio = torch.exp(lp - ol[idx])
                    pg = -torch.min(ratio * ad[idx],
                                   torch.clamp(ratio, 0.8, 1.2) * ad[idx]).mean()
                    vl = nn.MSELoss()(v.squeeze(-1), rt[idx])
                    loss = pg + 0.5*vl - ent_coeff*ent
                    opt.zero_grad()
                    loss.backward()
                    # FIX #10: clip ALL params including log_std
                    nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                    opt.step()
                    loss_val = pg.item()
            return loss_val

        al_pg = ppo_update(allied_model, allied_opt, allied_log_std,
                           al_o, al_a, al_l, al_adv, al_ret)
        # Axis gets 3x more PPO epochs + higher entropy for diverse strategies
        ax_pg = ppo_update(axis_model, axis_opt, axis_log_std,
                           ax_o, ax_a, ax_l, ax_adv, ax_ret,
                           epochs_override=num_epochs * 5, ent_coeff=0.05)

        # League snapshot — save Axis weights every 50 iterations
        if iteration > 0 and iteration % 50 == 0:
            league.append(copy.deepcopy(axis_model.state_dict()))
            log(f"  [League] Saved Axis snapshot #{len(league)} (iter {iteration})")

            # Activate league once we have 2+ snapshots
            if len(league) >= 2 and not use_league:
                use_league = True
                log(f"  [League] ACTIVATED — Allied now faces diverse opponents")
                log(f"  [League] Mix: 50% current Axis, 30% past snapshots, 20% random")

            # Load a random league snapshot into the league_model
            if len(league) >= 2:
                random_snapshot = league[np.random.randint(0, len(league))]
                league_model.load_state_dict(random_snapshot)
                league_model.eval()

        # Logging
        if iteration % 5 == 0:
            elapsed = time.time() - start_time
            sps = total_steps / max(elapsed, 1)
            rem = (total_iterations - iteration) / max(iteration, 1) * elapsed
            eta = f"{rem/60:.1f}m" if rem < 3600 else f"{rem/3600:.1f}h"
            wr = allied_wins / max(total_games, 1)

            # Timing breakdown
            t_total_iter = t_gpu + t_rust + t_numpy
            gpu_pct = t_gpu / max(t_total_iter, 1e-6) * 100
            rust_pct = t_rust / max(t_total_iter, 1e-6) * 100
            np_pct = t_numpy / max(t_total_iter, 1e-6) * 100

            log(f"I{iteration:4d}/{total_iterations} | {total_steps:>10,} steps | "
                f"{sps:>8,.0f} sps | ETA: {eta:>6s} | "
                f"Games: {total_games:>5d} | Allied: {wr:.0%} | "
                f"A_pg: {al_pg:.4f} X_pg: {ax_pg:.4f}")
            log(f"  Timing: GPU {gpu_pct:.0f}% ({t_gpu:.2f}s) | "
                f"Rust {rust_pct:.0f}% ({t_rust:.2f}s) | "
                f"NumPy {np_pct:.0f}% ({t_numpy:.2f}s)")

            log_timing({
                "iteration": iteration, "total_steps": total_steps,
                "sps": round(sps), "games": total_games,
                "allied_win_rate": round(wr, 3),
                "gpu_sec": round(t_gpu, 3), "rust_sec": round(t_rust, 3),
                "numpy_sec": round(t_numpy, 3),
                "gpu_pct": round(gpu_pct, 1), "rust_pct": round(rust_pct, 1),
                "league_size": len(league), "league_active": use_league,
            })

        # Save full resumable checkpoint
        if iteration % 100 == 0 and iteration > 0:
            torch.save({
                "allied": allied_model.state_dict(),
                "axis": axis_model.state_dict(),
                "allied_opt": allied_opt.state_dict(),
                "axis_opt": axis_opt.state_dict(),
                "iteration": iteration,
                "allied_wins": allied_wins,
                "axis_wins": axis_wins,
                "total_games": total_games,
                "total_steps": total_steps,
                "league": [s for s in league],  # all Axis snapshots
            }, save_path / f"selfplay_{iteration}.pt")
            log(f"  [Checkpoint] Saved iter {iteration} (resumable)")

    # Final
    torch.save({
        "allied": allied_model.state_dict(),
        "axis": axis_model.state_dict(),
        "allied_opt": allied_opt.state_dict(),
        "axis_opt": axis_opt.state_dict(),
        "iteration": total_iterations,
        "allied_wins": allied_wins,
        "axis_wins": axis_wins,
        "total_games": total_games,
        "total_steps": total_steps,
        "league": [s for s in league],
    }, save_path / "selfplay_final.pt")

    t = time.time() - start_time
    print(f"\nDone! {total_steps:,} steps in {t:.0f}s ({total_steps/t:,.0f} sps)")
    print(f"Games: {total_games} | Allied {allied_wins} ({allied_wins/max(total_games,1):.0%})")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--num-envs", type=int, default=128)
    p.add_argument("--iterations", type=int, default=500)
    p.add_argument("--steps-per-iter", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--save-dir", type=str, default="checkpoints_selfplay_v3")
    a = p.parse_args()

    print("=" * 70)
    print("  RL TripleA — Self-Play V3 (Rayon Parallel + MPS GPU)")
    print(f"  {a.num_envs} envs on 16 CPU cores (Rayon) + MPS GPU neural net")
    print(f"  Full mechanics: naval, AA, subs, BBs, bombing, capitals")
    print("=" * 70)

    train_selfplay(
        num_envs=a.num_envs,
        total_iterations=a.iterations,
        steps_per_iter=a.steps_per_iter,
        batch_size=a.batch_size,
        lr=a.lr,
        save_dir=a.save_dir,
    )
