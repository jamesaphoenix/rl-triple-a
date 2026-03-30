"""V2 Gymnasium environment with multi-phase action space.

Key differences from v1:
- Agent controls ALL phases: purchase, combat move, non-combat move, placement
- Axis opponent uses ProAI-inspired heuristics
- Action space decomposed by phase:
  - Purchase: MultiDiscrete (unit counts)
  - Combat Move: per-territory attack decisions (which territories to attack from where)
  - Non-Combat Move: per-territory reinforcement decisions
  - Placement: distribution of purchased units across factories
- Observation includes phase indicator
"""

from __future__ import annotations
from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .game_state import (
    GameState, ALL_PLAYERS, AXIS_PLAYERS, ALLIED_PLAYERS,
    PLAYER_INDEX,
)
from .units import (
    PURCHASABLE_UNITS, UNIT_TYPE_INDEX, NUM_UNIT_TYPES,
    UnitDomain,
)
from .engine import GameEngine
from .pro_ai import pro_purchase, pro_combat_moves, pro_noncombat_moves, pro_placement


# Phase IDs for observation encoding
PHASE_PURCHASE = 0
PHASE_COMBAT_MOVE = 1
PHASE_NONCOMBAT_MOVE = 2
PHASE_PLACEMENT = 3
NUM_PHASES = 4


class TripleAEnvV2(gym.Env):
    """Multi-phase environment where the agent controls all Allied decisions.

    The environment cycles through phases for each Allied player:
    1. Purchase -> agent picks units to buy
    2. Combat Move -> agent picks territory attacks
    3. Non-Combat Move -> agent picks reinforcements
    4. Placement -> agent distributes purchased units

    Action format varies by phase:
    - Purchase: array of (NUM_UNIT_TYPES,) unit counts
    - Combat Move: array of (num_territories,) attack scores per territory
      (top-K territories get attacked with available adjacent units)
    - Non-Combat Move: array of (num_territories,) reinforcement priority scores
    - Placement: array of (num_factories * NUM_UNIT_TYPES,) placement distribution
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        seed: int = 42,
        max_rounds: int = 15,
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        self.seed_val = seed
        self.max_rounds = max_rounds
        self.render_mode = render_mode

        self.engine = GameEngine(seed=seed)
        self.engine.max_rounds = max_rounds
        self.state = self.engine.state
        self.current_phase = PHASE_PURCHASE

        # We use a flat action space that covers the largest phase
        # The action is interpreted differently based on current_phase
        num_terr = self.state.num_territories

        # Action space: large enough for any phase
        # Purchase: first NUM_UNIT_TYPES entries (0-20 per unit)
        # Combat/NonCombat: num_territories scores (0-10 scale)
        # Placement: up to 10 factories * NUM_UNIT_TYPES
        action_dim = max(NUM_UNIT_TYPES * 21, num_terr, 10 * NUM_UNIT_TYPES)
        self.action_dim = action_dim

        self.observation_space = spaces.Box(
            low=-1.0, high=10.0,
            shape=(self.state.observation_size + NUM_PHASES + 1,),  # +phase one-hot +budget
            dtype=np.float32,
        )

        self.action_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(action_dim,),
            dtype=np.float32,
        )

        self._step_count = 0
        self._max_steps = max_rounds * len(ALL_PLAYERS) * NUM_PHASES * 2
        self._episode_reward = 0.0
        self._allied_player_queue: list[str] = []
        self._current_allied_player: str = ""

    def _get_obs(self) -> np.ndarray:
        """Get observation with phase indicator."""
        player = self._current_allied_player or ALLIED_PLAYERS[0]
        base_obs = self.state.to_observation(player)

        # Phase one-hot
        phase_onehot = np.zeros(NUM_PHASES, dtype=np.float32)
        phase_onehot[self.current_phase] = 1.0

        # Budget info
        p_idx = PLAYER_INDEX.get(player, 0)
        budget = np.array([self.state.pus[p_idx] / 50.0], dtype=np.float32)

        return np.concatenate([base_obs, phase_onehot, budget])

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed_val = seed
        self.engine = GameEngine(seed=self.seed_val)
        self.engine.max_rounds = self.max_rounds
        self.state = self.engine.state
        self.current_phase = PHASE_PURCHASE
        self._step_count = 0
        self._episode_reward = 0.0

        # Play Axis turns until first Allied player
        self._play_axis_turns()
        self._setup_allied_queue()

        obs = self._get_obs()
        info = self._make_info()
        return obs, info

    def _setup_allied_queue(self):
        """Set up the queue of Allied players for this round."""
        # Find Allied players in turn order
        self._allied_player_queue = [
            p for p in ALL_PLAYERS
            if p in ALLIED_PLAYERS and p == self.state.current_player
        ]
        if self._allied_player_queue:
            self._current_allied_player = self._allied_player_queue[0]
        else:
            self._current_allied_player = self.state.current_player

    def step(self, action: np.ndarray):
        self._step_count += 1
        s = self.state
        player = self._current_allied_player

        if s.game_over:
            return self._get_obs(), 0.0, True, False, self._make_info()

        reward = 0.0
        pre_income = sum(s.get_player_income(p) for p in ALLIED_PLAYERS)
        pre_vc = s.count_victory_cities("Allies")

        if self.current_phase == PHASE_PURCHASE:
            reward += self._execute_purchase(action, player)
            self.current_phase = PHASE_COMBAT_MOVE

        elif self.current_phase == PHASE_COMBAT_MOVE:
            reward += self._execute_combat_move(action, player)
            # Resolve battles automatically
            reward += self.engine.resolve_all_battles() * 0.01
            self.current_phase = PHASE_NONCOMBAT_MOVE

        elif self.current_phase == PHASE_NONCOMBAT_MOVE:
            reward += self._execute_noncombat_move(action, player)
            self.current_phase = PHASE_PLACEMENT

        elif self.current_phase == PHASE_PLACEMENT:
            reward += self._execute_placement(action, player)
            self.engine.end_turn()

            # Move to next player
            self.current_phase = PHASE_PURCHASE

            # Play any Axis turns
            self._play_axis_turns()

            # Check if game ended during Axis turns
            if not s.game_over:
                self._current_allied_player = s.current_player
            else:
                # Terminal reward
                post_income = sum(s.get_player_income(p) for p in ALLIED_PLAYERS)
                post_vc = s.count_victory_cities("Allies")
                reward += (post_vc - pre_vc) * 2.0
                reward += (post_income - pre_income) * 0.05

                if s.winner == "Allies":
                    reward += 100.0
                elif s.winner == "Axis":
                    reward -= 100.0

        # Income/VC rewards on phase transitions
        if self.current_phase == PHASE_PURCHASE and not s.game_over:
            post_income = sum(s.get_player_income(p) for p in ALLIED_PLAYERS)
            post_vc = s.count_victory_cities("Allies")
            reward += (post_vc - pre_vc) * 2.0
            reward += (post_income - pre_income) * 0.05

        self._episode_reward += reward

        terminated = s.game_over
        truncated = self._step_count >= self._max_steps

        if truncated and not terminated:
            axis_vc = s.count_victory_cities("Axis")
            allied_vc = s.count_victory_cities("Allies")
            reward += (allied_vc - axis_vc) * 5.0

        return self._get_obs(), reward, terminated, truncated, self._make_info()

    def _execute_purchase(self, action: np.ndarray, player: str) -> float:
        """Interpret action as purchase vector."""
        # Scale continuous action (0-1) to unit counts (0-20)
        purchase = np.clip(action[:NUM_UNIT_TYPES] * 20, 0, 20).astype(np.int32)
        self.engine.execute_purchase(purchase)
        return 0.0

    def _execute_combat_move(self, action: np.ndarray, player: str) -> float:
        """Interpret action as territory attack priorities.

        action[i] = priority score for attacking territory i.
        We attack the top territories where we have favorable odds.
        """
        p_idx = PLAYER_INDEX[player]
        alliance = "Axis" if player in AXIS_PLAYERS else "Allies"
        enemy_players = ALLIED_PLAYERS if alliance == "Axis" else AXIS_PLAYERS
        friendly_players = AXIS_PLAYERS if alliance == "Axis" else ALLIED_PLAYERS

        num_terr = self.state.num_territories
        scores = action[:num_terr]

        moves = []

        # Get territory indices sorted by attack priority
        sorted_indices = np.argsort(-scores)

        committed = np.zeros((num_terr, NUM_UNIT_TYPES), dtype=np.int32)

        for t_idx in sorted_indices:
            if scores[t_idx] < 0.3:  # threshold for "don't attack"
                break

            t = self.state.territories[t_idx]
            if t.is_impassable or t.is_water:
                continue
            if t.owner not in enemy_players:
                continue

            # Gather available attackers
            for n_idx in t.neighbor_indices:
                nt = self.state.territories[n_idx]
                if nt.is_impassable:
                    continue
                if nt.owner not in friendly_players and not nt.is_water:
                    continue

                available = nt.units[p_idx].copy() - committed[n_idx]
                available = np.maximum(available, 0)
                available[UNIT_TYPE_INDEX["factory"]] = 0
                available[UNIT_TYPE_INDEX["aaGun"]] = 0

                # Keep some defense
                if nt.production >= 2 and not nt.is_water:
                    inf_idx = UNIT_TYPE_INDEX["infantry"]
                    keep = min(1, int(available[inf_idx]))
                    available[inf_idx] = max(0, available[inf_idx] - keep)

                if np.sum(available) > 0:
                    moves.append((n_idx, t_idx, available))
                    committed[n_idx] += available

        self.engine.execute_combat_moves(moves)
        return 0.0

    def _execute_noncombat_move(self, action: np.ndarray, player: str) -> float:
        """Interpret action as territory reinforcement priorities."""
        p_idx = PLAYER_INDEX[player]
        alliance = "Axis" if player in AXIS_PLAYERS else "Allies"
        enemy_players = ALLIED_PLAYERS if alliance == "Axis" else AXIS_PLAYERS
        friendly_players = AXIS_PLAYERS if alliance == "Axis" else ALLIED_PLAYERS

        num_terr = self.state.num_territories
        scores = action[:num_terr]

        moves = []

        # Find territories to reinforce (high score = pull units toward here)
        sorted_targets = np.argsort(-scores)

        for target_idx in sorted_targets[:10]:  # top 10 reinforcement targets
            if scores[target_idx] < 0.3:
                break

            t = self.state.territories[target_idx]
            if t.is_impassable or t.is_water:
                continue
            if t.owner not in friendly_players:
                continue

            # Pull from adjacent lower-priority territories
            for n_idx in t.neighbor_indices:
                nt = self.state.territories[n_idx]
                if nt.is_impassable or nt.is_water:
                    continue
                if nt.owner not in friendly_players:
                    continue
                if scores[n_idx] >= scores[target_idx]:
                    continue  # don't pull from higher priority

                available = nt.units[p_idx].copy()
                available[UNIT_TYPE_INDEX["factory"]] = 0
                available[UNIT_TYPE_INDEX["aaGun"]] = 0

                # Only move land units
                for u_idx in range(NUM_UNIT_TYPES):
                    if PURCHASABLE_UNITS[u_idx].domain != UnitDomain.LAND:
                        available[u_idx] = 0

                # Keep minimum defense
                inf_idx = UNIT_TYPE_INDEX["infantry"]
                if nt.production > 0 and available[inf_idx] > 1:
                    available[inf_idx] -= 1

                if np.sum(available) > 0:
                    moves.append((n_idx, target_idx, available))

        self.engine.execute_noncombat_moves(moves)
        return 0.0

    def _execute_placement(self, action: np.ndarray, player: str) -> float:
        """Interpret action as placement distribution across factories."""
        p_idx = PLAYER_INDEX[player]
        factories = self.state.get_factories(player)

        if not factories:
            self.engine.pending_purchases = {}
            return 0.0

        remaining = dict(self.engine.pending_purchases)
        if not remaining:
            return 0.0

        # Separate land and sea
        land_units = {}
        sea_units = {}
        for u_idx, count in remaining.items():
            ut = PURCHASABLE_UNITS[u_idx]
            if ut.domain == UnitDomain.SEA:
                sea_units[u_idx] = count
            else:
                land_units[u_idx] = count

        placements = []

        # Use action to weight distribution across factories
        factory_weights = np.zeros(len(factories), dtype=np.float32)
        for i, f_idx in enumerate(factories):
            if i < len(action) - NUM_UNIT_TYPES:
                factory_weights[i] = action[NUM_UNIT_TYPES + i]
            else:
                factory_weights[i] = 1.0

        # Normalize weights
        total_w = factory_weights.sum()
        if total_w > 0:
            factory_weights /= total_w
        else:
            factory_weights = np.ones(len(factories)) / len(factories)

        # Distribute land units by weight
        for i, f_idx in enumerate(factories):
            t = self.state.territories[f_idx]
            capacity = t.factory_capacity if t.factory_capacity else t.production
            placed = 0
            units_array = np.zeros(NUM_UNIT_TYPES, dtype=np.int32)

            for u_idx in sorted(land_units.keys(),
                               key=lambda x: PURCHASABLE_UNITS[x].defense, reverse=True):
                count = land_units.get(u_idx, 0)
                alloc = max(1, int(count * factory_weights[i]))
                alloc = min(alloc, count, max(0, capacity - placed))
                if alloc > 0:
                    units_array[u_idx] = alloc
                    land_units[u_idx] -= alloc
                    if land_units[u_idx] == 0:
                        del land_units[u_idx]
                    placed += alloc

            if np.sum(units_array) > 0:
                placements.append((f_idx, units_array))

        # Place remaining land units at first factory with capacity
        for f_idx in factories:
            if not land_units:
                break
            t = self.state.territories[f_idx]
            capacity = t.factory_capacity if t.factory_capacity else t.production
            # Check how much already placed
            already = sum(p[1].sum() for p in placements if p[0] == f_idx)
            remaining_cap = capacity - already
            if remaining_cap <= 0:
                continue
            units_array = np.zeros(NUM_UNIT_TYPES, dtype=np.int32)
            for u_idx in list(land_units.keys()):
                count = land_units[u_idx]
                place = min(count, remaining_cap)
                if place > 0:
                    units_array[u_idx] = place
                    land_units[u_idx] -= place
                    remaining_cap -= place
                    if land_units[u_idx] == 0:
                        del land_units[u_idx]
            if np.sum(units_array) > 0:
                placements.append((f_idx, units_array))

        # Sea units in adjacent sea zone
        if sea_units:
            for f_idx in factories:
                t = self.state.territories[f_idx]
                for n_idx in t.neighbor_indices:
                    if self.state.territories[n_idx].is_water:
                        units_array = np.zeros(NUM_UNIT_TYPES, dtype=np.int32)
                        for u_idx, count in list(sea_units.items()):
                            units_array[u_idx] = count
                            del sea_units[u_idx]
                        if np.sum(units_array) > 0:
                            placements.append((n_idx, units_array))
                        break
                if not sea_units:
                    break

        self.engine.execute_placement(placements)
        return 0.0

    def _play_axis_turns(self):
        """Play Axis turns using ProAI heuristics."""
        while not self.state.game_over and self.state.current_player in AXIS_PLAYERS:
            player = self.state.current_player
            purchase = pro_purchase(self.state, player)
            self.engine.execute_purchase(purchase)

            combat_moves = pro_combat_moves(self.state, player)
            self.engine.execute_combat_moves(combat_moves)
            self.engine.resolve_all_battles()

            noncombat_moves = pro_noncombat_moves(self.state, player)
            self.engine.execute_noncombat_moves(noncombat_moves)

            placement = pro_placement(self.state, player, self.engine.pending_purchases)
            self.engine.execute_placement(placement)

            self.engine.end_turn()

    def _make_info(self) -> dict:
        s = self.state
        return {
            "current_player": self._current_allied_player,
            "phase": self.current_phase,
            "phase_name": ["purchase", "combat_move", "noncombat_move", "placement"][
                self.current_phase
            ],
            "round": s.round,
            "pus": int(s.pus[PLAYER_INDEX.get(self._current_allied_player, 0)]),
            "allied_vc": s.count_victory_cities("Allies"),
            "axis_vc": s.count_victory_cities("Axis"),
            "allied_income": sum(s.get_player_income(p) for p in ALLIED_PLAYERS),
            "axis_income": sum(s.get_player_income(p) for p in AXIS_PLAYERS),
            "winner": s.winner,
        }

    def render(self):
        if self.render_mode != "human":
            return
        s = self.state
        phase_names = ["Purchase", "Combat Move", "Non-Combat Move", "Placement"]
        print(f"\n=== Round {s.round} | {self._current_allied_player} | "
              f"{phase_names[self.current_phase]} ===")
        print(f"Allied VCs: {s.count_victory_cities('Allies')} | "
              f"Axis VCs: {s.count_victory_cities('Axis')}")
        for p in ALL_PLAYERS:
            p_idx = PLAYER_INDEX[p]
            income = s.get_player_income(p)
            print(f"  {p}: {s.pus[p_idx]} PUs, income={income}")
