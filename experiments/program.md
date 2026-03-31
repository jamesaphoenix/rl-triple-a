# Auto-Research Program — TripleA Allied Agent

You are an autonomous research agent optimising a reinforcement learning Allied agent for WW2v3 1942 (Axis & Allies). The agent must beat a human Axis player (James's dad) consistently.

**NEVER STOP.** The human may be away. Continue working indefinitely until manually stopped.

---

## Project Config

```
PROJECT_NAME: rl-triple-a
PRIMARY_METRIC: allied_win_rate (from checkpoints_phase2/timing.jsonl)
METRIC_DIRECTION: higher_is_better (but sweet spot is 55-70% against strong Axis)
EVAL_COMMAND: tail -1 checkpoints_phase2/timing.jsonl | python3 -c "import sys,json; d=json.loads(sys.stdin.read()); print(json.dumps(d))"
TIME_BUDGET_MINUTES: 20
```

### Architecture History

**V2 (original, RETIRED):** Flat MLP, 8.9M params on 16,215-dim input. 93% of params wasted in first layer projecting noisy input. Hit architectural ceiling at 81% after 2.7M games and 12 experiments. No hyperparameter change could break it.

**V3 (current):** GNN + Attention, 690k params. Territory-aware processing:
1. TerritoryEncoder: raw 100-dim per territory compressed to 37-dim, embedded to 128-dim
2. 3x GNN layers: multi-head attention over adjacency graph with skip connections. Each territory learns from its neighbors. After 3 layers, each territory knows its 3-hop neighborhood.
3. GlobalAttentionPool: 3 learned query tokens (purchase/attack/reinforce) attend to all territories
4. Separate action heads: purchase (from pooled), attack (per-territory), reinforce (per-territory)
5. Value head: from pooled global representation

**Why V3:** The Rust game engine outputs the same 16,215-dim observation. Only the PyTorch network changed. V3 matches the network structure to the problem structure (territories as graph nodes, adjacency as edges). 92% fewer params but 100% allocated to strategy learning.

**Training speed tradeoff:** V3 is slower per step (~360 sps vs 8,000 sps) because GNN attention is O(n^2) over territories. But it should learn better strategies per game, breaking the 81% ceiling.

### What The Agent Can Change

| Parameter / File | Location | What it controls |
|-----------------|----------|-----------------|
| Axis bid (PUs) | `train_phase2.py:30-32` | Extra starting PUs for Axis |
| Axis PPO epochs | `train_selfplay.py:291` | How many gradient steps Axis gets |
| Learning rate | `train_selfplay.py` or `train_phase2.py` | LR for training |
| Entropy coefficient | `train_selfplay.py` | Exploration vs exploitation |
| Reward shaping | `rust_engine/src/lib.rs` | How rewards are computed |
| GNN layers | `src/network.py` ActorCriticV3 | Depth of graph processing (currently 3) |
| Territory embed dim | `src/network.py` ActorCriticV3 | Embedding size (currently 128) |
| Attention heads | `src/network.py` ActorCriticV3 | Multi-head attention heads (currently 4) |
| Num envs | CLI arg `--num-envs` | Parallel games (currently 128) |
| Batch size | CLI arg `--batch-size` | PPO mini-batch size (currently 1024) |
| Steps per iteration | CLI arg `--steps-per-iter` | Rollout length (currently 256) |

### Performance Tuning (V3 specific)

If training is too slow (GPU bottlenecked at 90%+):
- Reduce num_envs from 128 to 32-64 (smaller batch = faster per step)
- Reduce GNN layers from 3 to 2 (less compute, 2-hop awareness)
- Reduce territory_embed_dim from 128 to 64 (halves attention compute)
- Reduce num_heads from 4 to 2

### What The Agent Must NOT Change

- The game rules in `rust_engine/src/lib.rs` — verified correct
- The Rust engine interface (observation/action format)
- The `hud/` web UI code
- National objectives definitions

### Current State

- **V3 architecture**: Training from scratch on Mac Studio (~360 sps)
- **V2 best model**: 2.7M games, 81% ceiling (available in checkpoints)
- **Goal**: Break the 81% ceiling with V3 architecture
- **Ultimate goal**: Allied agent that beats James's dad's Axis play

---

## Experiment Modes (adapted for RL)

### MACRO (Architecture/Algorithm Changes)
- Tune GNN depth, embed dim, attention heads
- Modify observation encoding in Rust
- Change reward function structure
- Add curriculum stages
- Try different algorithms (SAC instead of PPO)

### MICRO (Hyperparameter Tuning)
- Adjust bid amount (+/- PUs)
- Tune learning rates
- Adjust entropy coefficient
- Change batch size / rollout length / num_envs
- Modify reward coefficients

### GROWING_DATA (More Training)
- Run more iterations
- Add league diversity
- Performance tune for faster training

---

## Decision Thresholds

| Allied Win Rate | Assessment | Action |
|----------------|------------|--------|
| > 90% | Way too easy | Increase bid or Axis strength |
| 80-90% | Too easy | Increase bid +4 PUs |
| **55-70%** | **SWEET SPOT** | **Keep training, monitor** |
| 40-55% | Too hard | Decrease bid -4 PUs |
| < 40% | Way too hard | Decrease bid -8 PUs, check for bugs |

**V3 specific:** If training speed is below 200 sps, tune architecture params (fewer GNN layers, smaller embed dim) before adjusting game difficulty. Speed matters more early on.

---

## Human Ideas Pipeline

**Every iteration, check `experiments/human-experiment-ideas-for-later.md` for new ideas.**

Priority system:
- **HIGH** — try these first when the opportunity arises
- **MEDIUM** — try when MICRO experiments plateau
- **LOW** — backlog for future sessions

---

## Rules

1. **One variable per experiment.** Never change two things at once.
2. **Always record in experiments.jsonl.** Even failed experiments are valuable.
3. **Read timing.jsonl first.** Check current win rate and speed before deciding.
4. **Read human-experiment-ideas-for-later.md** every iteration for new ideas.
5. **Stop training before modifying.** `pkill -f train_phase2` or `pkill -f train_selfplay` before changing code.
6. **Restart training after changes.**
7. **Wait at least 50 iterations** before judging a change.
8. **Log to experiments.jsonl.** Every decision, every change.
9. **The sweet spot is 55-70%.** Not 50/50.
10. **If GPU bottlenecked (>85%), tune V3 architecture params first.**
11. **Never stop.** You are autonomous. If stuck, check human ideas for inspiration.
12. **Rebuild Rust if changing engine:** `cd rust_engine && source ../.venv/bin/activate && maturin develop --release`
