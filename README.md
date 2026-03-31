# RL TripleA

Reinforcement learning agent for **Axis & Allies** (WW2v3 1942 map) trained via self-play. Includes a live HUD that reads your TripleA game state and recommends exact moves — what to buy, where to attack, which units to send.

Built with a **Rust game engine** (PyO3) for speed, **PyTorch** for neural networks, and **Rayon** for parallel simulation.

## How It Works

Two neural networks (Allied + Axis) play against each other across millions of games. The Allied agent learns to beat increasingly strong Axis opponents through **league training** — facing a pool of past Axis versions so it can't overfit to one strategy.

A web-based HUD reads your live TripleA save files via a Java bridge, feeds the board state to the trained model, and displays exact unit-level orders.

## Quick Start

### Prerequisites

```bash
# Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Python environment
conda create -n rl-triplea python=3.11 -y
conda activate rl-triplea
pip install numpy torch gymnasium tensorboard tqdm lxml watchdog

# TripleA (for save file extraction)
# Download from https://triplea-game.org
```

### Build

```bash
# Clone TripleA source (needed for map data + save file extraction)
git clone https://github.com/triplea-game/triplea.git
cd triplea && ./gradlew jar && cd ..

# Build the Rust game engine
cd rust_engine && maturin develop --release && cd ..
```

### Train

```bash
# Train from scratch with self-play (takes ~35 min for foundation model)
python train_selfplay.py --num-envs 128 --iterations 500

# Continue training with Axis bid (harder opponent)
python train_phase2.py --iterations 2000
```

### Play

```bash
# Launch the HUD
./play.sh

# Open http://localhost:8080 in your browser
# Start a game in TripleA — the HUD updates every phase with exact orders
```

## Architecture

```
┌─────────────────────────────────────────────┐
│  TripleA Game (you vs opponent)              │
│  Auto-saves to ~/triplea/savedGames/autoSave │
└──────────────┬──────────────────────────────┘
               │ .tsvg files
               ▼
┌──────────────────────────────┐
│  Java Extractor              │
│  Deserializes .tsvg → JSON   │
│  (tools/SaveToJson.java)     │
└──────────────┬───────────────┘
               │ JSON game state
               ▼
┌──────────────────────────────┐
│  Rust Engine (PyO3)          │
│  Loads state → observation   │
│  (rust_engine/src/lib.rs)    │
└──────────────┬───────────────┘
               │ 16,215-dim observation
               ▼
┌──────────────────────────────┐
│  Neural Network (PyTorch)    │
│  ActorCritic, 8.9M params    │
│  (src/network.py)            │
└──────────────┬───────────────┘
               │ 337-dim action
               ▼
┌──────────────────────────────┐
│  Web HUD (localhost:8080)    │
│  Exact purchase/move/place   │
│  orders per phase            │
│  (hud/index.html)            │
└──────────────────────────────┘
```

## Project Structure

```
rl-triple-a/
├── rust_engine/           # Rust game engine (WW2v3 1942 rules)
│   └── src/lib.rs         #   Full rules: combat, naval, AA, subs, BBs, NOs, Chinese
├── src/
│   ├── network.py         #   ActorCriticV2 (MLP, 8.9M params)
│   ├── map_parser.py      #   Parse WW2v3 XML map definition
│   ├── game_data_export.py #  Export map data as numpy arrays for Rust
│   └── units.py           #   Unit type definitions
├── hud/
│   ├── server.py          #   HUD backend — watches saves, runs neural net
│   └── index.html         #   Web UI with phase tabs + exact orders
├── tools/
│   ├── SaveToJson.java    #   Java bridge: .tsvg → JSON
│   └── extract_live.sh    #   Shell wrapper for extraction
├── tests/
│   └── test_live_extraction.py  # Integration tests
├── experiments/
│   ├── program.md         #   Auto-research program definition
│   ├── prompt.md          #   Per-iteration research instructions
│   └── experiments.jsonl  #   Experiment log (29 experiments)
├── train_selfplay.py      #   Main training script (PPO self-play + leagues)
├── train_phase2.py        #   Phase 2: continue with Axis bid
├── play.sh                #   Launch HUD for playing
└── requirements.txt
```

## Game Rules Implemented

The Rust engine implements WW2v3 1942 rules verified against TripleA's Java source (24 bugs found and fixed via adversarial audit):

- Naval combat, amphibious assaults, transport capacity
- AA fire (1 shot per aircraft, hit on 1/6)
- Submarine first-strike (both sides, negated by destroyer)
- Battleship 2-hit
- Artillery support (1:1 infantry pairing, +1 attack)
- Transport casualties last
- Shore bombardment (capped at amphibious landing count)
- Capital capture (PU theft, income blocked)
- Chinese special rules (free infantry, movement restriction, no PU income)
- National objectives (17 NOs with allied exclusion + surface exclusion)
- Victory at 13 VCs (Projection of Power), checked per-round
- Blitz blocked by AA/factories
- Tech: OFF | National Objectives: ON

## Training Details

| Parameter | Value |
|-----------|-------|
| Algorithm | PPO with GAE |
| Training mode | Self-play + league (40% current / 40% past / 20% random) |
| Games trained | 2.7 million |
| Parallel environments | 128 (Rayon) |
| Observation | 16,215 dims |
| Action | 337 dims (13 purchase + 162 attack + 162 reinforce) |
| Network | 8.9M params, 3-layer MLP with LayerNorm |
| Game engine speed | ~5,000 steps/sec |
| Axis bid (training) | +60 PUs (stress test) |

## Auto-Research

The `experiments/` directory contains an autonomous research loop that monitors training metrics and adjusts hyperparameters. 29 experiments were conducted including bid adjustments, reward shaping, Axis training boosts, and league diversity — documented in `experiments.jsonl`.

## License

MIT
