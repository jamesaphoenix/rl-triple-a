#!/usr/bin/env python3
"""Entry point for V3 parallel training."""

import argparse
import sys
sys.path.insert(0, ".")

from src.train_v3 import train_v3, PPOConfigV3


def main():
    parser = argparse.ArgumentParser(description="Train Allied RL agent (V3 - parallel)")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--total-steps", type=int, default=2_000_000)
    parser.add_argument("--max-rounds", type=int, default=15)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--num-steps", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--save-dir", type=str, default="checkpoints_v3")
    parser.add_argument("--log-dir", type=str, default="runs_v3")
    args = parser.parse_args()

    config = PPOConfigV3(
        learning_rate=args.lr,
        num_envs=args.num_envs,
        total_timesteps=args.total_steps,
        max_rounds=args.max_rounds,
        hidden_size=args.hidden_size,
        num_steps=args.num_steps,
        batch_size=args.batch_size,
        save_dir=args.save_dir,
        log_dir=args.log_dir,
    )

    print("=" * 60)
    print("  RL TripleA V3 - Parallel Multi-Phase Training")
    print(f"  Parallel environments: {config.num_envs}")
    print(f"  Steps per rollout: {config.num_steps}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Total steps: {config.total_timesteps:,}")
    print("  Opponent: ProAI-inspired Axis heuristic")
    print("=" * 60)

    train_v3(config)


if __name__ == "__main__":
    main()
