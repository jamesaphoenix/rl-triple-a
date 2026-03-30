# Auto-Research Loop — One Iteration

Run ONE experiment cycle. You are fully autonomous — do NOT ask permission.

## Step 1: Read Context

1. Read `experiments/program.md` for rules and thresholds
2. Read `experiments/experiments.jsonl` for history (what's been tried)
3. Read `experiments/human-experiment-ideas-for-later.md` for queued ideas
4. Read current metrics: `tail -5 checkpoints_phase2/timing.jsonl`
5. Read training log: `tail -5 checkpoints_phase2/training.log`
6. Check if training is running: `ps aux | grep train_phase2 | grep -v grep | wc -l`

## Step 2: Assess Current State

Extract from timing.jsonl:
- `allied_win_rate` — the primary metric
- `games` — total games played this phase
- `sps` — training speed
- `iteration` — how far through training

Determine:
- Is training running? If not, restart it.
- Is win rate in sweet spot (55-70%)?
- Has enough data accumulated (>50 iterations since last change)?

## Step 3: Decide Action

Based on win rate and experiment history:

**If allied_win_rate > 80% and iteration > 50 since last change:**
→ Axis too weak. Pick ONE micro change:
  - Increase Axis bid by +4 PUs
  - OR increase Axis PPO epochs
  - OR decrease Allied learning rate

**If allied_win_rate < 55% and iteration > 50 since last change:**
→ Axis too strong. Pick ONE micro change:
  - Decrease Axis bid by -4 PUs
  - OR decrease Axis PPO epochs
  - OR increase Allied entropy for more exploration

**If allied_win_rate in 55-70%:**
→ Sweet spot! Check `experiments/human-experiment-ideas-for-later.md` for HIGH priority ideas.
  - **First time in sweet spot?** → Implement LEAGUE TRAINING (highest human priority)
  - Already have leagues? → Consider MACRO from human ideas (GNN, obs features, etc.)
  - No good ideas? → Just let it train more (GROWING_DATA)

**If training crashed or stopped:**
→ Restart with current settings. Log the crash.

## Step 4: Execute Change

1. Stop training: `pkill -f train_phase2`
2. Make ONE code change (edit the relevant file)
3. Rebuild Rust if needed: `cd rust_engine && maturin develop --release`
4. Clear old logs: `rm -f checkpoints_phase2/training.log checkpoints_phase2/timing.jsonl`
5. Restart training: `conda run -n rl-triplea python train_phase2.py --iterations 2000 --num-envs 128 &`
6. Wait 60 seconds, verify training started

## Step 5: Record

Append to `experiments/experiments.jsonl`:
```json
{
  "id": <next_id>,
  "timestamp": "<ISO 8601>",
  "hypothesis": "<what you expect to happen>",
  "variable": "<what you changed>",
  "old_value": "<before>",
  "new_value": "<after>",
  "type": "deterministic",
  "optimization_mode": "MICRO|MACRO|GROWING_DATA",
  "primary_metric": <current_allied_win_rate>,
  "primary_metric_prev": <previous_allied_win_rate>,
  "delta": "<change>",
  "decision": "keep|monitoring|discard|crash",
  "notes": "<reasoning>"
}
```

## Step 6: Report

Print a brief summary:
- Current state (win rate, games, iteration)
- What you changed and why
- What you expect to happen
- When to check again (next loop iteration)

## Reminders

- One variable per experiment
- Wait 50+ iterations before judging
- Sweet spot is 55-70% Allied win rate
- Never modify the foundation model
- Never modify game rules
- Always record in experiments.jsonl
- Never stop — you are autonomous
