# Human Experiment Ideas

Ideas for future experiments, added by James or auto-research agent.
Priority: HIGH items should be tried before MEDIUM/LOW.

## URGENT — Known Bugs

- **Land units can enter sea zones without transports** — the Rust engine doesn't enforce transport requirements for amphibious movement. The model learns illegal moves (e.g. "move infantry from Archangel → 4 Sea Zone"). Fix: check for transports in execute_combat when land units target sea zones. Mine TripleA's Java source at `game-app/game-core/src/main/java/games/strategy/triplea/delegate/move/validation/MoveValidator.java` for the exact transport validation logic.

- **Mine TripleA Java source for ALL rules** — the auto-research agent should systematically read the TripleA Java source code in `triplea/` and compare every rule against `rust_engine/src/lib.rs`. Key files to mine: MoveValidator.java, BattleDelegate.java, PlaceDelegate.java, TransportTracker.java, CasualtySelector.java. Fix every discrepancy found.

## HIGH Priority

- **Build TripleA bot (autopilot)** — implement TripleA's `AbstractAi` Java interface so the model plays directly inside TripleA as a player. Java class connects to Python model via localhost socket. No manual HUD needed. Key file: `triplea/game-app/game-core/src/main/java/games/strategy/triplea/ai/AbstractAi.java`. The bot would appear as a player option in TripleA's dropdown menu.

- **League training** — already implemented, keep improving diversity.

- **Opponent sampling diversity** — already implemented (40/40/20 mix).

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
