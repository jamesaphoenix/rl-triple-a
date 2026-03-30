#!/usr/bin/env python3
"""Entry point for V2 multi-phase training with ProAI opponent."""

import argparse
import sys
sys.path.insert(0, ".")

from src.train_v2 import train_v2, PPOConfigV2


def main():
    parser = argparse.ArgumentParser(description="Train Allied RL agent (V2 - all phases)")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--total-steps", type=int, default=2_000_000)
    parser.add_argument("--max-rounds", type=int, default=15)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--num-steps", type=int, default=256)
    parser.add_argument("--save-dir", type=str, default="checkpoints_v2")
    parser.add_argument("--log-dir", type=str, default="runs_v2")
    args = parser.parse_args()

    config = PPOConfigV2(
        learning_rate=args.lr,
        num_envs=args.num_envs,
        total_timesteps=args.total_steps,
        max_rounds=args.max_rounds,
        hidden_size=args.hidden_size,
        num_steps=args.num_steps,
        save_dir=args.save_dir,
        log_dir=args.log_dir,
    )

    print("=" * 60)
    print("  RL TripleA V2 - Multi-Phase Allied Agent")
    print("  Map: World War II v3 1942")
    print("  Agent: Allies (all phases - purchase/move/place)")
    print("  Opponent: Axis (ProAI-inspired heuristic)")
    print("=" * 60)
    print(f"\nConfig: {config}\n")

    train_v2(config)


if __name__ == "__main__":
    main()
