"""Gymnasium environment for WW2v3 1942.

The RL agent controls all Allied players (Russians, British, Chinese, Americans).
Axis players (Germans, Japanese, Italians) use a heuristic opponent policy.

Action space is decomposed into phases:
1. Purchase: continuous vector of unit quantities
2. Combat Move: select territories to attack and units to send
3. Placement: assign purchased units to factories

For training efficiency, we use a simplified action space:
- Purchase: MultiDiscrete over unit counts (bounded by budget)
- Movement: Flattened territory-pair selections
- Placement: Automatic (distribute to factories by capacity)
"""

from __future__ import annotations
from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .game_state import (
    GameState, ALL_PLAYERS, AXIS_PLAYERS, ALLIED_PLAYERS,
    PLAYER_INDEX, NUM_PLAYERS,
)
from .units import PURCHASABLE_UNITS, UNIT_TYPE_INDEX, NUM_UNIT_TYPES, UnitDomain
from .engine import (
    GameEngine, heuristic_purchase, random_combat_moves,
    random_noncombat_moves,
)
from .combat import estimate_battle_odds


class TripleAEnv(gym.Env):
    """Gymnasium environment for Allied play in WW2v3 1942.

    Observation: Flat vector encoding the full game state.
    Action: Purchase vector (how many of each unit to buy).

    Combat moves and placement are handled by heuristic sub-policies
    (can be upgraded to learned policies later).
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

        # Initialize to get dimensions
        self.engine = GameEngine(seed=seed)
        self.engine.max_rounds = max_rounds
        self.state = self.engine.state

        # Observation space
        obs_size = self.state.observation_size
        self.observation_space = spaces.Box(
            low=-1.0, high=10.0, shape=(obs_size,), dtype=np.float32
        )

        # Action space: purchase vector
        # Each unit type can be purchased 0-20 times (bounded by budget at runtime)
        self.action_space = spaces.MultiDiscrete(
            [21] * NUM_UNIT_TYPES, dtype=np.int32
        )

        self._step_count = 0
        self._max_steps = max_rounds * len(ALL_PLAYERS) * 2  # safety limit

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed_val = seed
        self.engine = GameEngine(seed=self.seed_val)
        self.engine.max_rounds = self.max_rounds
        self.state = self.engine.state
        self._step_count = 0

        # If first player is Axis, play their turns first
        self._play_axis_turns()

        obs = self.state.to_observation(self.state.current_player)
        info = {
            "current_player": self.state.current_player,
            "round": self.state.round,
            "pus": int(self.state.pus[self.state.current_player_idx]),
        }
        return obs, info

    def step(self, action: np.ndarray):
        """Take a step as the current Allied player.

        Args:
            action: Purchase vector of shape (NUM_UNIT_TYPES,)

        Returns:
            obs, reward, terminated, truncated, info
        """
        self._step_count += 1
        s = self.state
        player = s.current_player

        if s.game_over:
            obs = s.to_observation(player)
            return obs, 0.0, True, False, {"current_player": player}

        # Record state before turn for reward calculation
        pre_income = sum(s.get_player_income(p) for p in ALLIED_PLAYERS)
        pre_vc = s.count_victory_cities("Allies")

        # Execute Allied turn
        purchase = np.array(action, dtype=np.int32)
        self.engine.execute_purchase(purchase)

        # Combat moves via heuristic
        combat_moves = _smart_combat_moves(s, player)
        self.engine.execute_combat_moves(combat_moves)
        tuv_swing = self.engine.resolve_all_battles()

        # Non-combat: no moves for now
        # Placement: distribute to factories automatically
        self._auto_place(player)

        self.engine.end_turn()

        # Play all Axis turns
        self._play_axis_turns()

        # Calculate reward
        post_income = sum(s.get_player_income(p) for p in ALLIED_PLAYERS)
        post_vc = s.count_victory_cities("Allies")

        reward = 0.0
        reward += tuv_swing * 0.01  # TUV swing
        reward += (post_income - pre_income) * 0.1  # income change
        reward += (post_vc - pre_vc) * 1.0  # victory cities

        # Terminal rewards
        terminated = s.game_over
        if terminated:
            if s.winner == "Allies":
                reward += 100.0
            elif s.winner == "Axis":
                reward -= 100.0

        truncated = self._step_count >= self._max_steps
        if truncated and not terminated:
            # Score by victory cities
            axis_vc = s.count_victory_cities("Axis")
            allied_vc = s.count_victory_cities("Allies")
            reward += (allied_vc - axis_vc) * 5.0

        obs = s.to_observation(s.current_player if not s.game_over else ALLIED_PLAYERS[0])
        info = {
            "current_player": s.current_player,
            "round": s.round,
            "pus": int(s.pus[s.current_player_idx]) if not s.game_over else 0,
            "allied_vc": s.count_victory_cities("Allies"),
            "axis_vc": s.count_victory_cities("Axis"),
            "allied_income": post_income,
            "winner": s.winner,
        }

        return obs, reward, terminated, truncated, info

    def _play_axis_turns(self):
        """Play all consecutive Axis player turns using heuristics."""
        while not self.state.game_over and self.state.current_player in AXIS_PLAYERS:
            player = self.state.current_player
            purchase = heuristic_purchase(self.state, player)
            self.engine.execute_purchase(purchase)

            combat_moves = _smart_combat_moves(self.state, player)
            self.engine.execute_combat_moves(combat_moves)
            self.engine.resolve_all_battles()

            self._auto_place(player)
            self.engine.end_turn()

    def _auto_place(self, player: str):
        """Automatically place purchased units at factories."""
        p_idx = PLAYER_INDEX[player]
        factories = self.state.get_factories(player)
        if not factories:
            # If no factories but have capital, treat capital as factory
            for t in self.state.territories:
                if t.is_capital and t.capital_of == player and t.owner == player:
                    factories = [t.index]
                    break

        if not factories:
            self.engine.pending_purchases = {}
            return

        remaining = dict(self.engine.pending_purchases)
        placements = []

        # Separate land and sea units
        land_units = {}
        sea_units = {}
        for u_idx, count in remaining.items():
            ut = PURCHASABLE_UNITS[u_idx]
            if ut.domain == UnitDomain.SEA:
                sea_units[u_idx] = count
            else:
                land_units[u_idx] = count

        # Place land units at factories (spread across factories)
        for factory_idx in factories:
            t = self.state.territories[factory_idx]
            capacity = t.factory_capacity if t.factory_capacity else t.production
            placed = 0

            units_array = np.zeros(NUM_UNIT_TYPES, dtype=np.int32)
            for u_idx, count in list(land_units.items()):
                can_place = min(count, max(0, capacity - placed))
                if can_place > 0:
                    units_array[u_idx] = can_place
                    land_units[u_idx] -= can_place
                    if land_units[u_idx] == 0:
                        del land_units[u_idx]
                    placed += can_place

            placements.append((factory_idx, units_array))

        # Place sea units in adjacent sea zones of factories
        for factory_idx in factories:
            t = self.state.territories[factory_idx]
            for n_idx in t.neighbor_indices:
                nt = self.state.territories[n_idx]
                if nt.is_water:
                    units_array = np.zeros(NUM_UNIT_TYPES, dtype=np.int32)
                    for u_idx, count in list(sea_units.items()):
                        if count > 0:
                            units_array[u_idx] = count
                            del sea_units[u_idx]
                    if np.sum(units_array) > 0:
                        placements.append((n_idx, units_array))
                    break

        self.engine.execute_placement(placements)

    def render(self):
        if self.render_mode != "human":
            return
        s = self.state
        print(f"\n=== Round {s.round} | {s.current_player}'s turn ===")
        print(f"Allied VCs: {s.count_victory_cities('Allies')} | "
              f"Axis VCs: {s.count_victory_cities('Axis')}")
        for p in ALL_PLAYERS:
            p_idx = PLAYER_INDEX[p]
            income = s.get_player_income(p)
            print(f"  {p}: {s.pus[p_idx]} PUs, income={income}")


def _smart_combat_moves(
    state: GameState, player: str
) -> list[tuple[int, int, np.ndarray]]:
    """Smarter combat movement heuristic.

    Prioritizes attacking weakly defended enemy territories with high production.
    """
    p_idx = PLAYER_INDEX[player]
    alliance = "Axis" if player in AXIS_PLAYERS else "Allies"
    enemy_players = ALLIED_PLAYERS if alliance == "Axis" else AXIS_PLAYERS
    friendly_players = AXIS_PLAYERS if alliance == "Axis" else ALLIED_PLAYERS
    rng = np.random.default_rng()
    moves = []

    # Find attack opportunities
    for t in state.territories:
        if t.is_impassable or t.is_water:
            continue
        if t.owner not in enemy_players:
            continue

        # Sum enemy defenders
        dfn = np.zeros(NUM_UNIT_TYPES, dtype=np.int32)
        for ep in enemy_players:
            if ep in PLAYER_INDEX:
                dfn += t.units[PLAYER_INDEX[ep]]

        dfn_strength = sum(dfn[i] * PURCHASABLE_UNITS[i].defense
                          for i in range(NUM_UNIT_TYPES)
                          if not PURCHASABLE_UNITS[i].is_factory
                          and not PURCHASABLE_UNITS[i].is_aa)

        if dfn_strength == 0:
            # Undefended territory - send 1 unit
            for n_idx in t.neighbor_indices:
                nt = state.territories[n_idx]
                if nt.is_water or nt.is_impassable:
                    continue
                if nt.owner in friendly_players:
                    for u_idx in range(NUM_UNIT_TYPES):
                        ut = PURCHASABLE_UNITS[u_idx]
                        if ut.is_factory or ut.is_aa or ut.domain != UnitDomain.LAND:
                            continue
                        if nt.units[p_idx, u_idx] > 1:
                            units = np.zeros(NUM_UNIT_TYPES, dtype=np.int32)
                            units[u_idx] = 1
                            moves.append((nt.index, t.index, units))
                            break
                    break
            continue

        # Gather potential attackers from neighboring friendly territories
        total_atk = np.zeros(NUM_UNIT_TYPES, dtype=np.int32)
        sources = []
        for n_idx in t.neighbor_indices:
            nt = state.territories[n_idx]
            if nt.is_impassable:
                continue
            if nt.owner in friendly_players or nt.is_water:
                available = nt.units[p_idx].copy()
                # Don't move factories or AA
                factory_idx = UNIT_TYPE_INDEX["factory"]
                aa_idx = UNIT_TYPE_INDEX["aaGun"]
                available[factory_idx] = 0
                available[aa_idx] = 0
                # Keep at least 1 unit for defense
                for u_idx in range(NUM_UNIT_TYPES):
                    if available[u_idx] > 1:
                        available[u_idx] -= 1
                    else:
                        available[u_idx] = 0
                if np.sum(available) > 0:
                    total_atk += available
                    sources.append((n_idx, available))

        atk_strength = sum(total_atk[i] * PURCHASABLE_UNITS[i].attack
                          for i in range(NUM_UNIT_TYPES)
                          if not PURCHASABLE_UNITS[i].is_factory
                          and not PURCHASABLE_UNITS[i].is_aa)

        # Attack if we have significant advantage (1.5x strength or more)
        if atk_strength > dfn_strength * 1.5 and t.production >= 1:
            for src_idx, available in sources:
                moves.append((src_idx, t.index, available))

    return moves
