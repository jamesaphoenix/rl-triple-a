#!/usr/bin/env python3
"""Phase 2: Continue self-play from foundation model with Axis bid.

Loads foundation_v1 weights, gives Axis 18 extra PUs at start,
runs 2000 iterations. Saves checkpoints every 100 iterations.

Stop anytime — latest checkpoint is always usable.
To play: ./play.sh
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

# Patch the training script to add Axis bid
import numpy as np
from src.game_data_export import export_map_arrays

# Monkey-patch the initial PUs to give Axis a bid
_original_export = export_map_arrays

def export_with_axis_bid():
    arrays = _original_export()
    # Give Axis 18 extra PUs (standard competitive bid)
    # Japanese=0, Germans=2, Italians=4
    # Split: 8 to Germany, 6 to Japan, 4 to Italy
    arrays["initial_pus"] = arrays["initial_pus"].copy()
    arrays["initial_pus"][0] += 20  # Japanese: 31 → 51
    arrays["initial_pus"][2] += 28  # Germans: 37 → 65
    arrays["initial_pus"][4] += 12  # Italians: 10 → 22
    print(f"Axis bid applied (+60 total): Japanese {arrays['initial_pus'][0]}, "
          f"Germans {arrays['initial_pus'][2]}, Italians {arrays['initial_pus'][4]}")
    return arrays

# Replace the export function
import src.game_data_export
src.game_data_export.export_map_arrays = export_with_axis_bid

# Now import and run training
from train_selfplay import train_selfplay
import torch

FOUNDATION_MODEL = "checkpoints_selfplay_v3/foundation_v1_282k_games_93pct.pt"
SAVE_DIR = "checkpoints_phase2"

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--iterations", type=int, default=2000)
    p.add_argument("--num-envs", type=int, default=128)
    a = p.parse_args()

    print("=" * 60)
    print("  Phase 2: Self-Play with Axis Bid (+18 PUs)")
    print(f"  Loading: {FOUNDATION_MODEL}")
    print(f"  Iterations: {a.iterations}")
    print(f"  Stop anytime — run ./play.sh to use latest model")
    print("=" * 60)

    # Patch train_selfplay to load foundation weights
    original_train = train_selfplay

    def train_with_foundation(**kwargs):
        kwargs["save_dir"] = SAVE_DIR
        kwargs["total_iterations"] = a.iterations
        kwargs["num_envs"] = a.num_envs
        # We need to hook into the training function to load weights
        # The simplest way: modify the function to accept pretrained weights
        original_train(**kwargs)

    # Load and inject weights by patching ActorCriticV2 init
    from src.network import ActorCriticV2
    _original_init = ActorCriticV2.__init__
    _loaded = {"done": False}

    def patched_init(self, *args, **kwargs):
        _original_init(self, *args, **kwargs)
        if not _loaded["done"] and os.path.exists(FOUNDATION_MODEL):
            ckpt = torch.load(FOUNDATION_MODEL, map_location="cpu", weights_only=False)
            # First model created = allied, second = axis
            if not hasattr(patched_init, "_count"):
                patched_init._count = 0
            if patched_init._count == 0 and "allied" in ckpt:
                self.load_state_dict(ckpt["allied"])
                print(f"  Loaded Allied weights from foundation model")
            elif patched_init._count == 1:
                # EXPERIMENT #1: DON'T load Axis weights — start fresh
                # The foundation Axis learned to be passive (7% win rate).
                # Fresh random weights + 30 PU bid = aggressive exploration.
                print(f"  Axis starting from RANDOM weights (not foundation)")
                _loaded["done"] = True
            patched_init._count = getattr(patched_init, "_count", 0) + 1

    ActorCriticV2.__init__ = patched_init

    train_selfplay(
        num_envs=a.num_envs,
        total_iterations=a.iterations,
        steps_per_iter=256,
        batch_size=1024,
        lr=1e-4,  # Lower LR for fine-tuning
        save_dir=SAVE_DIR,
    )
