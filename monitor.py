#!/usr/bin/env python3
"""Auto-monitor: checks win rate every 30 min, adjusts Axis bid if needed.

Strategy:
- Allied win rate 55-70%: PERFECT — both sides learning, keep going
- Allied win rate 40-55%: Axis too strong — reduce bid by 4 PUs, restart
- Allied win rate < 40%: Axis dominating — reduce bid by 8 PUs, restart
- Allied win rate > 80%: Allied coasting — increase bid by 4 PUs, restart
- Allied win rate > 90%: Way too easy — increase bid by 8 PUs, restart

The sweet spot is 55-70%. That means Allied is being challenged but still
winning more than half — it's learning robust strategies under pressure.

Usage:
    python monitor.py

Runs forever. Ctrl+C to stop. Logs to checkpoints_phase2/monitor.log
"""

import sys
import os
import json
import time
import subprocess
import signal
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

CHECK_INTERVAL = 1800  # 30 minutes
TIMING_LOG = "checkpoints_phase2/timing.jsonl"
MONITOR_LOG = "checkpoints_phase2/monitor.log"
TRAIN_SCRIPT = "/Users/jamesaphoenix/Desktop/rl-triple-a/train_phase2.py"

# Bid adjustment thresholds
SWEET_SPOT_LOW = 0.55
SWEET_SPOT_HIGH = 0.70
TOO_HARD_LOW = 0.40
TOO_EASY_HIGH = 0.80
WAY_TOO_EASY = 0.90

# Current bid state
BID_FILE = "checkpoints_phase2/current_bid.json"


def log(msg):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line, flush=True)
    with open(MONITOR_LOG, "a") as f:
        f.write(line + "\n")


def get_current_bid():
    if os.path.exists(BID_FILE):
        with open(BID_FILE) as f:
            return json.load(f)
    return {"japanese": 6, "germans": 8, "italians": 4, "total": 18}


def save_bid(bid):
    with open(BID_FILE, "w") as f:
        json.dump(bid, f, indent=2)


def get_recent_win_rate(lookback_entries=20):
    """Get Allied win rate from recent training log entries."""
    if not os.path.exists(TIMING_LOG):
        return None, 0

    entries = []
    with open(TIMING_LOG) as f:
        for line in f:
            try:
                entries.append(json.loads(line.strip()))
            except:
                pass

    if len(entries) < 5:
        return None, 0

    recent = entries[-lookback_entries:]
    latest = recent[-1]
    return latest.get("allied_win_rate", None), latest.get("games", 0)


def is_training_running():
    result = subprocess.run(
        ["pgrep", "-f", "train_phase2"],
        capture_output=True, text=True
    )
    return result.returncode == 0


def kill_training():
    log("Stopping current training...")
    subprocess.run(["pkill", "-f", "train_phase2"], capture_output=True)
    time.sleep(5)
    subprocess.run(["pkill", "-9", "-f", "train_phase2"], capture_output=True)
    time.sleep(2)


def start_training(bid):
    """Start Phase 2 training with given bid."""
    save_bid(bid)

    # Modify the train_phase2.py bid values via environment variable
    env = os.environ.copy()
    env["AXIS_BID_JAPANESE"] = str(bid["japanese"])
    env["AXIS_BID_GERMANS"] = str(bid["germans"])
    env["AXIS_BID_ITALIANS"] = str(bid["italians"])

    log(f"Starting training with bid: Japan +{bid['japanese']}, "
        f"Germany +{bid['germans']}, Italy +{bid['italians']} "
        f"(total +{bid['total']})")

    subprocess.Popen(
        ["conda", "run", "-n", "rl-triplea", "python", TRAIN_SCRIPT,
         "--iterations", "2000", "--num-envs", "128"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(10)


def adjust_bid(current_bid, allied_win_rate):
    """Adjust bid based on win rate. Returns new bid or None if no change needed."""
    total = current_bid["total"]

    if allied_win_rate >= WAY_TOO_EASY:
        # Way too easy — big increase
        delta = 8
        log(f"Allied at {allied_win_rate:.0%} — WAY TOO EASY. Increasing bid by {delta}")
    elif allied_win_rate >= TOO_EASY_HIGH:
        # Too easy — small increase
        delta = 4
        log(f"Allied at {allied_win_rate:.0%} — too easy. Increasing bid by {delta}")
    elif allied_win_rate < TOO_HARD_LOW:
        # Way too hard — big decrease
        delta = -8
        log(f"Allied at {allied_win_rate:.0%} — WAY TOO HARD. Decreasing bid by {delta}")
    elif allied_win_rate < SWEET_SPOT_LOW:
        # Too hard — small decrease
        delta = -4
        log(f"Allied at {allied_win_rate:.0%} — too hard. Decreasing bid by {delta}")
    else:
        # Sweet spot
        log(f"Allied at {allied_win_rate:.0%} — IN SWEET SPOT ({SWEET_SPOT_LOW:.0%}-{SWEET_SPOT_HIGH:.0%}). No change.")
        return None

    new_total = max(0, total + delta)
    # Distribute proportionally: ~45% Germany, ~33% Japan, ~22% Italy
    new_bid = {
        "germans": max(0, round(new_total * 0.45)),
        "japanese": max(0, round(new_total * 0.33)),
        "italians": max(0, new_total - round(new_total * 0.45) - round(new_total * 0.33)),
        "total": new_total,
    }
    return new_bid


def main():
    os.makedirs("checkpoints_phase2", exist_ok=True)

    log("=" * 60)
    log("  Auto-Monitor Started")
    log(f"  Check interval: {CHECK_INTERVAL // 60} minutes")
    log(f"  Sweet spot: {SWEET_SPOT_LOW:.0%} - {SWEET_SPOT_HIGH:.0%} Allied win rate")
    log("=" * 60)

    while True:
        try:
            win_rate, games = get_recent_win_rate()
            bid = get_current_bid()

            if win_rate is None:
                log(f"No data yet. Training running: {is_training_running()}")
                if not is_training_running():
                    log("Training not running — starting with current bid")
                    start_training(bid)
            else:
                log(f"Check: Allied {win_rate:.0%} | Games: {games:,} | "
                    f"Bid: +{bid['total']} | Training: {is_training_running()}")

                if not is_training_running():
                    log("Training stopped — restarting")
                    start_training(bid)
                elif win_rate < SWEET_SPOT_LOW or win_rate > TOO_EASY_HIGH:
                    new_bid = adjust_bid(bid, win_rate)
                    if new_bid:
                        kill_training()
                        time.sleep(5)
                        start_training(new_bid)
                    else:
                        log("No adjustment needed")
                else:
                    log("In sweet spot — continuing")

            log(f"Next check in {CHECK_INTERVAL // 60} minutes...")
            time.sleep(CHECK_INTERVAL)

        except KeyboardInterrupt:
            log("Monitor stopped by user")
            break
        except Exception as e:
            log(f"Error: {e}")
            time.sleep(60)


if __name__ == "__main__":
    main()
