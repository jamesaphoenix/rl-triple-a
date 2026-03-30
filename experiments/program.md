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

### What The Agent Can Change

| Parameter / File | Location | What it controls |
|-----------------|----------|-----------------|
| Axis bid (PUs) | `train_phase2.py:30-32` | Extra starting PUs for Axis — makes game harder for Allies |
| Axis PPO epochs | `train_selfplay_v3.py:291` | How many gradient steps Axis gets (currently 3x Allied) |
| Allied learning rate | `train_phase2.py:92` | LR for fine-tuning (currently 1e-4) |
| Entropy coefficient | `train_selfplay_v3.py:278` | Exploration vs exploitation (currently 0.02) |
| Reward shaping | `rust_engine/src/lib.rs:248-257` | How rewards are computed (TUV swing, VC change, income) |
| Network hidden size | `src/network_v2.py:11` | Network capacity (currently 512) |
| Batch size | CLI arg `--batch-size` | PPO mini-batch size (currently 1024) |
| Steps per iteration | CLI arg `--steps-per-iter` | Rollout length (currently 256) |
| Observation encoding | `rust_engine/src/lib.rs` `get_observation()` | What the agent sees |

### What The Agent Must NOT Change

- The game rules in `rust_engine/src/lib.rs` (combat resolution, AA fire, etc.) — these are verified correct
- The `foundation_v1_282k_games_93pct.pt` checkpoint — this is the base model
- The `hud/` web UI code
- National objectives definitions
- The `DATA_DICTIONARY.md` and `MODEL_CARD.md`

### Current State

- **Foundation model**: 282k games, 93% Allied win rate (against co-trained Axis)
- **Phase 2 running**: +30 PU Axis bid, 3x Axis PPO epochs, loading foundation weights
- **Goal**: Get Allied win rate to 55-70% sweet spot against a strong Axis
- **Ultimate goal**: Allied agent recommends moves that beat James's dad's Axis play

---

## Experiment Modes (adapted for RL)

### MACRO (Architecture/Algorithm Changes)
- Change network architecture (add attention, GNN, skip connections)
- Modify observation encoding (add factory damage, add distance features)
- Change reward function structure
- Add curriculum stages
- Modify exploration strategy

### MICRO (Hyperparameter Tuning)
- Adjust bid amount (+/- 4 PUs)
- Tune learning rates
- Adjust entropy coefficient
- Change batch size / rollout length
- Modify reward coefficients (TUV weight, VC weight, income weight)

### GROWING_DATA (More Training)
- Run more iterations
- Increase num_envs for more parallel games
- Add league diversity (more snapshot frequency)

---

## Decision Thresholds

| Allied Win Rate | Assessment | Action |
|----------------|------------|--------|
| > 90% | Way too easy | Increase bid +8 PUs or increase Axis epochs |
| 80-90% | Too easy | Increase bid +4 PUs |
| **55-70%** | **SWEET SPOT** | **Keep training, monitor** |
| 40-55% | Too hard | Decrease bid -4 PUs or decrease Axis epochs |
| < 40% | Way too hard | Decrease bid -8 PUs, check for bugs |

---

## Files

```
experiments/
  program.md                              # This file
  prompt.md                               # Per-iteration instructions
  experiments.jsonl                        # Experiment log
  human-experiment-ideas-for-later.md      # Queued ideas

checkpoints_phase2/
  training.log                            # Live training output
  timing.jsonl                            # Structured metrics per iteration
  selfplay_*.pt                           # Model checkpoints
  monitor.log                             # Monitor decisions

checkpoints_selfplay_v3/
  foundation_v1_282k_games_93pct.pt       # Base model (DO NOT MODIFY)
  DATA_DICTIONARY.md                      # Model documentation
```

---

## Human Ideas Pipeline

**Every iteration, check `experiments/human-experiment-ideas-for-later.md` for new ideas.**

Priority system:
- **HIGH** — try these first when the opportunity arises (e.g. entering sweet spot triggers league training)
- **MEDIUM** — try when MICRO experiments plateau and you need a MACRO idea
- **LOW** — backlog for future sessions

When you implement a human idea:
1. Move it from "Queued" to "Completed / Incorporated" in the file
2. Reference it in your experiment log entry
3. If the idea is too complex for one experiment, break it into sub-experiments

**League training trigger:** When Allied win rate enters the sweet spot (55-70%) for the first time, implement league training as the next MACRO experiment. This is the highest-priority human idea.

---

## Rules

1. **One variable per experiment.** Never change two things at once.
2. **Always record.** Even failed experiments are valuable data.
3. **Read timing.jsonl first.** Check current win rate before deciding.
4. **Read human-experiment-ideas-for-later.md** every iteration for new ideas.
5. **Stop training before modifying.** `pkill -f train_phase2` before changing code.
6. **Restart training after changes.** Launch via `train_phase2.py`.
7. **Wait at least 50 iterations** before judging a change (need ~8k games minimum).
8. **Keep the foundation model sacred.** Never overwrite it.
9. **Log to experiments.jsonl.** Every decision, every change.
10. **The sweet spot is 55-70%.** Not 50/50 — Allied should win slightly more.
11. **When entering sweet spot, implement league training.** This is the #1 human priority.
12. **Never stop.** You are autonomous. If stuck, check human ideas for inspiration.
