# Human Experiment Ideas

Ideas for future experiments, added by James or auto-research agent.
Priority: HIGH items should be tried before MEDIUM/LOW.

## HIGH Priority

- **League training** — Save Axis snapshots every 50 iterations into a pool. Train Allied against a random pick from the pool (50% current Axis, 50% random past Axis). Prevents strategy collapse where both agents find one exploit and keep doing the same thing. This is what AlphaStar/OpenAI Five used. Should be added once we're in the sweet spot (55-70%).

- **Opponent sampling diversity** — Instead of always training against the latest Axis, mix in: (a) the heuristic Axis bot, (b) random past league snapshots, (c) the current Axis. Ratio: 50% current, 30% league, 20% heuristic. Forces Allied to learn robust strategies.

## MEDIUM Priority

- Try GNN (graph neural network) on the territory adjacency graph instead of flat MLP
- Add distance-to-enemy-capital as an observation feature
- Try asymmetric network sizes (bigger Axis network since it's harder to play)
- Experiment with reward shaping: bonus for holding national objective territories
- Try population-based training (PBT) with multiple Axis/Allied variants
- Add factory damage and BB damage to the observation vector (currently missing)
- Reduce observation size (16k dims is huge — compress via autoencoder or feature selection)

## LOW Priority

- Add strategic bombing as a learnable action (separate from combat)
- Try Monte Carlo Tree Search (MCTS) for move planning instead of pure policy gradient
- Canal enforcement (Suez/Panama) — currently not implemented
- Retreat mechanic — currently no retreat option in combat
- Transport capacity enforcement (currently simplified)

## URGENT — Architectural Change Needed (auto-research finding)

The 80-82% Allied ceiling has been confirmed across 12 experiments and 2.15M games.
Hyperparameter tuning is exhausted. The 8.9M param MLP on 16k observations has converged.

**Recommended first change: Observation compression (16k → ~2k dims)**
- 15 territories are impassable (zero information) — remove them
- 65 sea zones mostly empty — compress to "has enemy navy" flags
- Per-territory unit encoding (91 dims × 162) → only encode territories within 2 hops of frontline
- First layer drops from 8.3M params to ~1M — remaining capacity focuses on strategy
- Requires: changes to Rust `get_observation()` + Python `ActorCriticV2` input size
- Estimated implementation: 30-60 min

**Alternative: Double hidden layer size (512 → 1024)**
- Simpler change but doesn't address the noise problem
- Model goes from 8.9M to ~35M params
- May overfit faster without observation compression

## Completed / Incorporated
- ~~League training~~ — Done in Exp #7, 50% current + 30% league + 20% random
- ~~Axis 3x VC reward~~ — Done in Exp #4, dropped Allied from 83% to 74%
- ~~Random Axis weights~~ — Done in Exp #1
- ~~Higher Axis entropy~~ — Done in Exp #2
- ~~Axis 5x PPO epochs~~ — Done in Exp #6
