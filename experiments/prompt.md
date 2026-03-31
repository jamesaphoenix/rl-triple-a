# Auto-Research Loop — One Iteration

Run ONE experiment cycle. You are fully autonomous. Do NOT ask permission.

## Step 1: Read Context

1. Read `experiments/program.md` for rules, thresholds, and architecture info
2. Read `experiments/experiments.jsonl` for history
3. Read `experiments/human-experiment-ideas-for-later.md` for queued ideas
4. Read current metrics: `tail -5 checkpoints_phase2/timing.jsonl`
5. Read training log: `tail -5 checkpoints_phase2/training.log`
6. Check if training is running: `ps aux | grep train | grep python | grep -v grep | wc -l`

## Step 2: Assess Current State

Extract from timing.jsonl:
- `allied_win_rate` — the primary metric
- `games` — total games played
- `sps` — training speed (steps per second)
- `gpu_pct` — GPU utilization
- `iteration` — how far through training

Key questions:
- Is training running? If not, restart it.
- Is win rate in sweet spot (55-70%)?
- **Is GPU bottlenecked (>85%)?** If so, tune V3 architecture BEFORE adjusting difficulty.
- Has enough data accumulated (>50 iterations since last change)?

## Step 3: Decide Action

**If GPU > 85% AND sps < 300:**
The V3 GNN architecture is too heavy. Pick ONE:
  - Reduce num_envs from 128 to 64 or 32
  - Reduce GNN layers from 3 to 2
  - Reduce territory_embed_dim from 128 to 64
  - Reduce num_heads from 4 to 2

**If sps > 300 AND allied_win_rate > 80%:**
Axis too weak. Pick ONE:
  - Increase Axis bid by +4 PUs
  - Increase Axis PPO epochs
  - Decrease Allied learning rate

**If sps > 300 AND allied_win_rate < 55%:**
Axis too strong. Pick ONE:
  - Decrease Axis bid by -4 PUs
  - Decrease Axis PPO epochs

**If allied_win_rate in 55-70%:**
Sweet spot! Consider:
  - Just let it train more (GROWING_DATA)
  - Check human ideas for MACRO experiments

**If training crashed or stopped:**
Restart with current settings. Log the crash.

## Step 4: Execute Change

1. Stop training: `pkill -f train_selfplay` or `pkill -f train_phase2`
2. Make ONE code change
3. Rebuild Rust if needed: `cd rust_engine && source ../.venv/bin/activate && maturin develop --release`
4. Clear old logs if starting fresh
5. Restart training: `source .venv/bin/activate && python train_selfplay.py --num-envs 128 --iterations 2000 --save-dir checkpoints_phase2 &`
6. Wait 60 seconds, verify training started

## Step 5: Record

Append to `experiments/experiments.jsonl`:
```json
{
  "id": <next_id>,
  "timestamp": "<ISO 8601>",
  "hypothesis": "<what you expect>",
  "variable": "<what you changed>",
  "old_value": "<before>",
  "new_value": "<after>",
  "type": "deterministic",
  "optimization_mode": "MICRO|MACRO|GROWING_DATA",
  "primary_metric": <current_win_rate>,
  "primary_metric_prev": <previous>,
  "delta": "<change>",
  "decision": "keep|monitoring|discard|crash",
  "notes": "<reasoning>"
}
```

## Step 6: Report

Print a brief summary:
- Current state (win rate, games, iteration, sps, GPU%)
- What you changed and why
- What you expect to happen
- When to check again

## Reminders

- One variable per experiment
- Wait 50+ iterations before judging
- Sweet spot is 55-70% Allied win rate
- **V3 architecture: if GPU > 85%, tune architecture first, not game difficulty**
- Never modify game rules in Rust
- Always record in experiments.jsonl
- Never stop. You are autonomous.
- Use .venv not conda: `source .venv/bin/activate`
