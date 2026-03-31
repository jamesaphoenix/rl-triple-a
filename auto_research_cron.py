#!/usr/bin/env python3
"""Standalone auto-research script for cron execution.

Reads program.md thresholds, checks training metrics,
makes one adjustment if needed, logs to experiments.jsonl.

Run via cron every 30 minutes:
  */30 * * * * cd /path/to/rl-triple-a && .venv/bin/python auto_research_cron.py
"""

import sys
import os
import json
import time
import subprocess
import glob
from pathlib import Path

PROJECT = Path(__file__).parent
TIMING_LOG = PROJECT / "checkpoints_v3" / "timing.jsonl"
TRAINING_LOG = PROJECT / "checkpoints_v3" / "training.log"
EXPERIMENTS = PROJECT / "experiments" / "experiments.jsonl"
PROGRESS = PROJECT / "experiments" / "progress.txt"

def log_progress(msg):
    timestamp = time.strftime("[%Y-%m-%d %H:%M]")
    line = f"{timestamp} {msg}"
    print(line)
    with open(PROGRESS, "a") as f:
        f.write(line + "\n")
    # Trim to last 500 lines
    try:
        lines = PROGRESS.read_text().splitlines()
        if len(lines) > 500:
            PROGRESS.write_text("\n".join(lines[-500:]) + "\n")
    except:
        pass

def get_latest_metrics():
    if not TIMING_LOG.exists():
        return None
    try:
        last_line = TIMING_LOG.read_text().strip().split("\n")[-1]
        return json.loads(last_line)
    except:
        return None

def is_training_running():
    result = subprocess.run(["pgrep", "-f", "train_selfplay"], capture_output=True)
    return result.returncode == 0

def get_next_experiment_id():
    if not EXPERIMENTS.exists():
        return 0
    lines = [l for l in EXPERIMENTS.read_text().strip().split("\n") if l.strip()]
    if not lines:
        return 0
    try:
        last = json.loads(lines[-1])
        return last.get("id", 0) + 1
    except:
        return len(lines)

def record_experiment(data):
    with open(EXPERIMENTS, "a") as f:
        f.write(json.dumps(data) + "\n")

def restart_training(num_envs=64, iterations=2000, save_dir="checkpoints_v3"):
    subprocess.run(["pkill", "-f", "train_selfplay"], capture_output=True)
    time.sleep(3)
    cmd = (
        f"cd {PROJECT} && source .venv/bin/activate && "
        f"nohup python train_selfplay.py "
        f"--num-envs {num_envs} --iterations {iterations} "
        f"--steps-per-iter 256 --batch-size 512 "
        f"--save-dir {save_dir} "
        f"> training_output.log 2>&1 &"
    )
    subprocess.run(["bash", "-c", cmd])
    time.sleep(5)
    return is_training_running()

def main():
    log_progress("=" * 60)
    log_progress("AUTO-RESEARCH CHECK")

    # Check if training is running
    running = is_training_running()
    metrics = get_latest_metrics()

    if not running:
        log_progress("Training NOT running. Restarting...")
        if restart_training():
            log_progress("Training restarted successfully (64 envs, nohup)")
            record_experiment({
                "id": get_next_experiment_id(),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "hypothesis": "Auto-restart: training had stopped",
                "variable": "none (restart)",
                "old_value": "stopped", "new_value": "running",
                "type": "deterministic", "optimization_mode": "GROWING_DATA",
                "primary_metric": metrics["allied_win_rate"] if metrics else None,
                "primary_metric_prev": None, "delta": "n/a",
                "decision": "keep",
                "notes": "Cron auto-restart. Training was not running."
            })
        else:
            log_progress("ERROR: Could not restart training!")
        return

    if not metrics:
        log_progress("Training running but no metrics yet. Waiting.")
        return

    wr = metrics["allied_win_rate"]
    sps = metrics["sps"]
    gpu = metrics.get("gpu_pct", 0)
    games = metrics["games"]
    iteration = metrics["iteration"]

    log_progress(f"Iter {iteration} | {games:,} games | Allied {wr:.0%} | "
                 f"{sps:,} sps | GPU {gpu:.0f}%")

    # Decision logic from program.md
    action = None

    # GPU bottleneck check (V3 specific)
    if gpu > 85 and sps < 300:
        log_progress(f"GPU bottlenecked at {gpu:.0f}%, {sps} sps. "
                     "Consider reducing num_envs or GNN layers.")
        action = "gpu_bottleneck_warning"

    # Win rate thresholds
    elif wr > 0.90:
        log_progress(f"Allied {wr:.0%} — WAY TOO EASY")
        action = "too_easy"
    elif wr > 0.80:
        log_progress(f"Allied {wr:.0%} — too easy")
        action = "slightly_easy"
    elif 0.55 <= wr <= 0.70:
        log_progress(f"Allied {wr:.0%} — IN SWEET SPOT! Keep training.")
        action = "sweet_spot"
    elif 0.40 <= wr < 0.55:
        log_progress(f"Allied {wr:.0%} — too hard")
        action = "too_hard"
    elif wr < 0.40:
        log_progress(f"Allied {wr:.0%} — WAY TOO HARD")
        action = "way_too_hard"
    else:
        log_progress(f"Allied {wr:.0%} — borderline. Monitoring.")
        action = "monitoring"

    record_experiment({
        "id": get_next_experiment_id(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "hypothesis": f"Cron check: Allied {wr:.0%}, {sps} sps, GPU {gpu:.0f}%",
        "variable": "none (monitoring)",
        "old_value": "n/a", "new_value": "n/a",
        "type": "deterministic", "optimization_mode": "GROWING_DATA",
        "primary_metric": wr, "primary_metric_prev": wr, "delta": "0.0",
        "decision": "keep",
        "notes": f"Iter {iteration}, {games:,} games, {action}. "
                 f"Speed: {sps} sps, GPU: {gpu:.0f}%"
    })

    log_progress(f"Action: {action}")
    log_progress("Next check in 30 minutes.")

if __name__ == "__main__":
    main()
