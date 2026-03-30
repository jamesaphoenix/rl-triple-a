#!/usr/bin/env python3
"""Entry point for JAX GPU-accelerated training."""

import argparse
import sys
import os
sys.path.insert(0, ".")
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# Force CPU — JAX Metal has limited op support
# JIT+vmap on CPU is still 10-50x faster than pure Python
os.environ["JAX_PLATFORMS"] = "cpu"

from jax_src.train_jax import train


def main():
    parser = argparse.ArgumentParser(description="JAX RL training for TripleA")
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--total-steps", type=int, default=2_000_000)
    parser.add_argument("--num-steps", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--save-dir", type=str, default="checkpoints_jax")
    args = parser.parse_args()

    print("=" * 60)
    print("  RL TripleA — JAX GPU Training")
    print(f"  Parallel games: {args.num_envs}")
    print(f"  Total steps: {args.total_steps:,}")
    print("=" * 60)

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
