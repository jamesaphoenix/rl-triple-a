#!/usr/bin/env python3
"""Entry point for training the Allied RL agent."""

import argparse
import sys
sys.path.insert(0, ".")

from src.train import train, PPOConfig


def main():
    parser = argparse.ArgumentParser(description="Train Allied RL agent for WW2v3 1942")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--num-envs", type=int, default=8, help="Parallel environments")
    parser.add_argument("--total-steps", type=int, default=1_000_000, help="Total training steps")
    parser.add_argument("--max-rounds", type=int, default=15, help="Max game rounds")
    parser.add_argument("--hidden-size", type=int, default=512, help="Hidden layer size")
    parser.add_argument("--save-dir", type=str, default="checkpoints", help="Save directory")
    parser.add_argument("--log-dir", type=str, default="runs", help="Tensorboard log directory")
    args = parser.parse_args()

    config = PPOConfig(
        learning_rate=args.lr,
        num_envs=args.num_envs,
        total_timesteps=args.total_steps,
        max_rounds=args.max_rounds,
        hidden_size=args.hidden_size,
        save_dir=args.save_dir,
        log_dir=args.log_dir,
    )

    print("=" * 60)
    print("  RL TripleA - Allied Agent Training")
    print("  Map: World War II v3 1942")
    print("  You: Allies (Russians, British, Chinese, Americans)")
    print("  Opponent: Axis (Germans, Japanese, Italians) - Heuristic")
    print("=" * 60)
    print(f"\nConfig: {config}")
    print()

    train(config)


if __name__ == "__main__":
    main()
