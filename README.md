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

### Play Against the AI (Multiplayer Bot)

The RL bot runs as a local HTTP server. TripleA's `RLBot` Java class sends game
state each phase and the server returns purchase/move/place orders using the
trained neural net.

```bash
# 1. Start the action server (Allied model by default)
conda activate rl-triplea
python bot/action_server.py

# Or play as Axis:
RL_SIDE=axis python bot/action_server.py

# Or use a specific checkpoint:
RL_MODEL_PATH=checkpoints_phase2/selfplay_1800.pt python bot/action_server.py
```

The server listens on `http://localhost:8080`. Verify it's running:

```bash
curl http://localhost:8080/health
# {"status":"ok","model_loaded":true,"model_path":"...","side":"allied"}
```

**In TripleA:**
1. Host a new game on the **WW2v3 1942** map
2. Assign one side to **RLBot** (the AI)
3. Your opponent (e.g. your father) joins the hosted game
4. Play — the bot makes decisions via the action server each phase

### Run from a different machine (e.g. laptop → Mac Studio)

If the action server runs on your Mac Studio and you play TripleA on your laptop:

```bash
# On Mac Studio — bind to all interfaces so your laptop can reach it
conda activate rl-triplea
python -c "
import uvicorn
from bot.action_server import app
uvicorn.run(app, host='0.0.0.0', port=8080)
"

# On laptop — update RLBot.java's ACTION_SERVER_URL to point to Mac Studio:
#   private static final String ACTION_SERVER_URL = "http://Jamess-Mac-Studio.local:8080/api/action";
# Then rebuild TripleA with the updated RLBot.
```

### Run the Integration Tests

```bash
conda activate rl-triplea
python tests/test_bot_integration.py
# 43 tests: heuristic policy, model inference, API endpoints, multi-round stress
```

### Launch the HUD (recommendation mode — no bot, just suggestions)

```bash
./play.sh

# Open http://localhost:8080 in your browser
# Start a game in TripleA — the HUD updates every phase with exact orders
```

## Architecture

```
                        ┌───────────────────────────────────────┐
                        │  TripleA Game (WW2v3 1942)            │
                        │  You + friend vs AI  (or HUD mode)    │
                        └───────┬───────────────┬───────────────┘
                    (Bot mode)  │               │  (HUD mode)
                   RLBot.java   │               │  .tsvg auto-saves
                                │               │
               ┌────────────────▼──┐   ┌────────▼──────────────┐
               │  Action Server    │   │  Java Extractor       │
               │  POST /api/action │   │  .tsvg → JSON         │
               │  (bot/)           │   │  (tools/)             │
               └────────┬─────────┘   └────────┬──────────────┘
                        │                       │
                        ▼                       ▼
               ┌──────────────────────────────────────┐
               │  Rust Engine (PyO3)                   │
               │  Game state → 16,701-dim observation  │
               │  (rust_engine/src/lib.rs)             │
               └────────────────┬─────────────────────┘
                                │
                                ▼
               ┌──────────────────────────────────────┐
               │  Neural Network (PyTorch)             │
               │  ActorCriticV3 — GNN + Attention      │
               │  337-dim action (purchase/attack/reinforce) │
               │  (src/network.py)                     │
               └────────────────┬─────────────────────┘
                                │
                ┌───────────────┼───────────────┐
                ▼                               ▼
  ┌──────────────────────┐        ┌──────────────────────┐
  │  Bot: JSON response  │        │  HUD: Web UI         │
  │  → RLBot.java        │        │  localhost:8080       │
  │  executes moves      │        │  (hud/index.html)    │
  └──────────────────────┘        └──────────────────────┘
```

## Project Structure

```
rl-triple-a/
├── rust_engine/           # Rust game engine (WW2v3 1942 rules)
│   └── src/lib.rs         #   Full rules: combat, naval, AA, subs, BBs, NOs, Chinese
├── bot/
│   ├── action_server.py   #   FastAPI server — real model inference for RLBot
│   └── RLBot.java         #   TripleA AI player class (calls action server)
├── src/
│   ├── network.py         #   ActorCriticV2 (MLP) + ActorCriticV3 (GNN)
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
│   ├── test_live_extraction.py  # Live extraction pipeline tests
│   └── test_bot_integration.py  # Bot action server tests (43 tests)
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
| Observation | 16,701 dims |
| Action | 337 dims (13 purchase + 162 attack + 162 reinforce) |
| Network | ActorCriticV3 — GNN + Attention (~118K params) |
| Game engine speed | ~5,000 steps/sec |
| Axis bid (training) | +60 PUs (stress test) |

## Auto-Research

The `experiments/` directory contains an autonomous research loop that monitors training metrics and adjusts hyperparameters. 29 experiments were conducted including bid adjustments, reward shaping, Axis training boosts, and league diversity — documented in `experiments.jsonl`.

## License

MIT
