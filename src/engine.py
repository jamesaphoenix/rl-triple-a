"""Game engine implementing the WW2v3 1942 turn sequence.

Simplified for RL training:
- Handles purchase, combat movement, battle resolution, non-combat movement, placement
- Technology research is skipped (can be added later)
- Chinese special rules simplified
"""

from __future__ import annotations
from collections import defaultdict
from typing import Optional

import numpy as np

from .game_state import (
    GameState, Territory, ALL_PLAYERS, AXIS_PLAYERS, ALLIED_PLAYERS,
    PLAYER_INDEX, NUM_PLAYERS,
)
from .units import (
    PURCHASABLE_UNITS, UNIT_TYPE_INDEX, NUM_UNIT_TYPES,
    UnitDomain, INFANTRY, TRANSPORT,
)
from .combat import resolve_combat


class GameEngine:
    """Manages game flow and action execution."""

    def __init__(self, state: Optional[GameState] = None, seed: int = 42):
        self.state = state if state is not None else GameState()
        self.rng = np.random.default_rng(seed)
        self.pending_purchases: dict[int, int] = {}  # unit_type_idx -> count
        self.max_rounds = 15  # end game after 15 rounds

    def execute_purchase(self, purchase_vector: np.ndarray) -> float:
        """Execute a purchase action.

        Args:
            purchase_vector: Array of shape (NUM_UNIT_TYPES,) with counts to buy.

        Returns:
            Reward signal (negative if invalid purchase).
        """
        s = self.state
        player = s.current_player
        p_idx = s.current_player_idx
        budget = int(s.pus[p_idx])

        # Validate and cap purchases
        total_cost = 0
        valid_purchase = {}
        for i, count in enumerate(purchase_vector):
            count = max(0, int(count))
            if count == 0:
                continue
            ut = PURCHASABLE_UNITS[i]
            cost = ut.cost * count
            if total_cost + cost <= budget:
                valid_purchase[i] = count
                total_cost += cost
            else:
                # Buy as many as we can afford
                affordable = (budget - total_cost) // ut.cost
                if affordable > 0:
                    valid_purchase[i] = affordable
                    total_cost += ut.cost * affordable
                break

        self.pending_purchases = valid_purchase
        s.pus[p_idx] -= total_cost
        return 0.0

    def execute_combat_moves(self, moves: list[tuple[int, int, np.ndarray]]) -> float:
        """Execute combat movement.

        Args:
            moves: List of (from_territory, to_territory, units_to_move)
                  where units_to_move is array of shape (NUM_UNIT_TYPES,)

        Returns:
            Reward signal.
        """
        s = self.state
        p_idx = s.current_player_idx

        for from_idx, to_idx, units in moves:
            from_t = s.territories[from_idx]
            to_t = s.territories[to_idx]

            # Validate: territory must be adjacent
            if to_idx not in from_t.neighbor_indices:
                continue

            # Validate: player must have these units
            for u_idx in range(NUM_UNIT_TYPES):
                count = min(int(units[u_idx]), int(from_t.units[p_idx, u_idx]))
                if count > 0:
                    from_t.units[p_idx, u_idx] -= count
                    to_t.units[p_idx, u_idx] += count

        return 0.0

    def resolve_all_battles(self) -> float:
        """Resolve all battles where opposing forces coexist.

        Returns:
            Net TUV reward (positive = good for current player).
        """
        s = self.state
        p_idx = s.current_player_idx
        player = s.current_player
        alliance = s.current_alliance

        friendly_players = AXIS_PLAYERS if alliance == "Axis" else ALLIED_PLAYERS
        enemy_players = ALLIED_PLAYERS if alliance == "Axis" else AXIS_PLAYERS

        total_tuv_swing = 0.0

        for t in s.territories:
            if t.is_impassable:
                continue

            # Sum friendly and enemy units
            atk_units = np.zeros(NUM_UNIT_TYPES, dtype=np.int32)
            dfn_units = np.zeros(NUM_UNIT_TYPES, dtype=np.int32)

            for fp in friendly_players:
                if fp in PLAYER_INDEX:
                    atk_units += t.units[PLAYER_INDEX[fp]]
            for ep in enemy_players:
                if ep in PLAYER_INDEX:
                    dfn_units += t.units[PLAYER_INDEX[ep]]

            # Check if there's a battle (both sides have units)
            atk_combat = sum(atk_units[i] for i in range(NUM_UNIT_TYPES)
                            if not PURCHASABLE_UNITS[i].is_factory
                            and not PURCHASABLE_UNITS[i].is_aa)
            dfn_combat = sum(dfn_units[i] for i in range(NUM_UNIT_TYPES)
                            if not PURCHASABLE_UNITS[i].is_factory
                            and not PURCHASABLE_UNITS[i].is_aa)

            if atk_combat == 0 or dfn_combat == 0:
                continue

            result = resolve_combat(atk_units, dfn_units, rng=self.rng)
            total_tuv_swing += result.defender_losses_tuv - result.attacker_losses_tuv

            if result.attacker_wins:
                # Distribute remaining units back to current player
                t.units[p_idx] = result.attacker_remaining
                # Remove all enemy units
                for ep in enemy_players:
                    if ep in PLAYER_INDEX:
                        t.units[PLAYER_INDEX[ep]] = np.zeros(NUM_UNIT_TYPES, dtype=np.int32)
                # Remove other friendly player units (simplified - they contributed)
                for fp in friendly_players:
                    if fp != player and fp in PLAYER_INDEX:
                        t.units[PLAYER_INDEX[fp]] = np.zeros(NUM_UNIT_TYPES, dtype=np.int32)

                # Conquer territory
                if not t.is_water:
                    t.owner = player
                    # Capture enemy factory
                    factory_idx = UNIT_TYPE_INDEX["factory"]
                    for ep in enemy_players:
                        if ep in PLAYER_INDEX:
                            if t.units[PLAYER_INDEX[ep], factory_idx] > 0:
                                t.units[p_idx, factory_idx] += t.units[PLAYER_INDEX[ep], factory_idx]
                                t.units[PLAYER_INDEX[ep], factory_idx] = 0

            elif result.defender_wins:
                # Remove all attacker units
                for fp in friendly_players:
                    if fp in PLAYER_INDEX:
                        # Keep factories and AA guns
                        factory_idx = UNIT_TYPE_INDEX["factory"]
                        aa_idx = UNIT_TYPE_INDEX["aaGun"]
                        saved_factories = t.units[PLAYER_INDEX[fp], factory_idx]
                        saved_aa = t.units[PLAYER_INDEX[fp], aa_idx]
                        t.units[PLAYER_INDEX[fp]] = np.zeros(NUM_UNIT_TYPES, dtype=np.int32)
                        if t.owner == fp:
                            t.units[PLAYER_INDEX[fp], factory_idx] = saved_factories
                            t.units[PLAYER_INDEX[fp], aa_idx] = saved_aa

                # Distribute remaining to the territory owner
                if t.owner and t.owner in enemy_players and t.owner in PLAYER_INDEX:
                    t.units[PLAYER_INDEX[t.owner]] = result.defender_remaining
            else:
                # Draw - both sides eliminated
                for pi in range(NUM_PLAYERS):
                    factory_idx = UNIT_TYPE_INDEX["factory"]
                    aa_idx = UNIT_TYPE_INDEX["aaGun"]
                    saved_f = t.units[pi, factory_idx]
                    saved_aa = t.units[pi, aa_idx]
                    t.units[pi] = np.zeros(NUM_UNIT_TYPES, dtype=np.int32)
                    if ALL_PLAYERS[pi] == t.owner:
                        t.units[pi, factory_idx] = saved_f
                        t.units[pi, aa_idx] = saved_aa

        return total_tuv_swing

    def execute_noncombat_moves(self, moves: list[tuple[int, int, np.ndarray]]) -> float:
        """Execute non-combat movement. Same format as combat moves."""
        return self.execute_combat_moves(moves)  # same logic, different legality

    def execute_placement(self, placements: list[tuple[int, np.ndarray]]) -> float:
        """Place purchased units at factories.

        Args:
            placements: List of (factory_territory_idx, units_to_place)
                       where units_to_place is array of (NUM_UNIT_TYPES,)
        """
        s = self.state
        p_idx = s.current_player_idx
        player = s.current_player

        remaining = dict(self.pending_purchases)

        for factory_idx, units in placements:
            t = s.territories[factory_idx]
            if t.owner != player:
                continue

            # Check factory exists
            f_idx = UNIT_TYPE_INDEX["factory"]
            if t.units[p_idx, f_idx] == 0 and not t.is_capital:
                continue

            # Get factory capacity
            capacity = t.factory_capacity if t.factory_capacity else t.production
            placed = 0

            for u_idx in range(NUM_UNIT_TYPES):
                count = min(int(units[u_idx]), remaining.get(u_idx, 0))
                if count <= 0:
                    continue

                ut = PURCHASABLE_UNITS[u_idx]
                # Sea units go in adjacent sea zone
                if ut.domain == UnitDomain.SEA:
                    # Find adjacent sea zone
                    for n_idx in t.neighbor_indices:
                        nt = s.territories[n_idx]
                        if nt.is_water:
                            nt.units[p_idx, u_idx] += count
                            remaining[u_idx] = remaining.get(u_idx, 0) - count
                            break
                else:
                    if placed + count <= capacity:
                        t.units[p_idx, u_idx] += count
                        remaining[u_idx] = remaining.get(u_idx, 0) - count
                        placed += count
                    else:
                        can_place = max(0, capacity - placed)
                        if can_place > 0:
                            t.units[p_idx, u_idx] += can_place
                            remaining[u_idx] = remaining.get(u_idx, 0) - can_place
                            placed += can_place

        self.pending_purchases = {}
        return 0.0

    def end_turn(self) -> float:
        """End current player's turn, collect income, advance to next player.

        Returns:
            Reward signal.
        """
        s = self.state
        p_idx = s.current_player_idx
        player = s.current_player

        # Collect income
        income = s.get_player_income(player)
        s.pus[p_idx] += income

        # Advance to next player
        player_idx_in_order = ALL_PLAYERS.index(player)
        next_idx = (player_idx_in_order + 1) % len(ALL_PLAYERS)
        s.current_player_idx = PLAYER_INDEX[ALL_PLAYERS[next_idx]]

        # Check for new round
        if next_idx == 0:
            s.round += 1

        # Check victory
        winner = s.check_victory()
        if winner:
            s.game_over = True
            s.winner = winner

        if s.round > self.max_rounds:
            s.game_over = True
            # Determine winner by victory cities
            axis_vc = s.count_victory_cities("Axis")
            allied_vc = s.count_victory_cities("Allies")
            s.winner = "Axis" if axis_vc > allied_vc else "Allies"

        s.phase = "purchase"
        return 0.0

    def play_full_turn(
        self,
        purchase_fn,
        combat_move_fn,
        noncombat_move_fn,
        placement_fn,
    ) -> float:
        """Play a complete turn for the current player using provided decision functions.

        Each function takes (GameState, player) and returns the appropriate action.

        Returns total reward for the turn.
        """
        s = self.state
        player = s.current_player

        # Purchase phase
        s.phase = "purchase"
        purchase = purchase_fn(s, player)
        reward = self.execute_purchase(purchase)

        # Combat move phase
        s.phase = "combat_move"
        combat_moves = combat_move_fn(s, player)
        reward += self.execute_combat_moves(combat_moves)

        # Battle phase
        s.phase = "battle"
        reward += self.resolve_all_battles()

        # Non-combat move phase
        s.phase = "noncombat_move"
        noncombat_moves = noncombat_move_fn(s, player)
        reward += self.execute_noncombat_moves(noncombat_moves)

        # Placement phase
        s.phase = "place"
        placements = placement_fn(s, player)
        reward += self.execute_placement(placements)

        # End turn
        s.phase = "end_turn"
        reward += self.end_turn()

        return reward


def random_purchase(state: GameState, player: str) -> np.ndarray:
    """Random purchase strategy - buy random affordable units."""
    p_idx = PLAYER_INDEX[player]
    budget = int(state.pus[p_idx])
    purchase = np.zeros(NUM_UNIT_TYPES, dtype=np.int32)

    rng = np.random.default_rng()
    for _ in range(20):  # try 20 random purchases
        u_idx = rng.integers(0, NUM_UNIT_TYPES)
        ut = PURCHASABLE_UNITS[u_idx]
        if ut.is_factory or ut.is_aa:
            continue
        if ut.cost <= budget:
            purchase[u_idx] += 1
            budget -= ut.cost

    return purchase


def random_combat_moves(state: GameState, player: str) -> list[tuple[int, int, np.ndarray]]:
    """Random combat movement - move random units to random adjacent territories."""
    p_idx = PLAYER_INDEX[player]
    alliance = "Axis" if player in AXIS_PLAYERS else "Allies"
    enemy_players = ALLIED_PLAYERS if alliance == "Axis" else AXIS_PLAYERS
    rng = np.random.default_rng()
    moves = []

    for t in state.territories:
        if t.is_impassable:
            continue
        total_units = np.sum(t.units[p_idx])
        if total_units == 0:
            continue

        # Find enemy neighbors
        enemy_neighbors = []
        for n_idx in t.neighbor_indices:
            nt = state.territories[n_idx]
            if nt.owner in enemy_players and not nt.is_impassable:
                enemy_neighbors.append(n_idx)

        if enemy_neighbors and rng.random() > 0.5:
            target = rng.choice(enemy_neighbors)
            # Move half of combat units
            units = np.zeros(NUM_UNIT_TYPES, dtype=np.int32)
            for u_idx in range(NUM_UNIT_TYPES):
                ut = PURCHASABLE_UNITS[u_idx]
                if ut.is_factory or ut.is_aa:
                    continue
                count = int(t.units[p_idx, u_idx])
                move_count = count // 2
                if move_count > 0:
                    units[u_idx] = move_count
            if np.sum(units) > 0:
                moves.append((t.index, target, units))

    return moves


def random_noncombat_moves(state: GameState, player: str) -> list[tuple[int, int, np.ndarray]]:
    """Random non-combat movement."""
    return []  # No movement for simplicity


def random_placement(state: GameState, player: str) -> list[tuple[int, np.ndarray]]:
    """Place all purchased units at the capital factory."""
    factories = state.get_factories(player)
    if not factories:
        return []

    # All pending purchases go to the first factory (capital usually)
    return []  # Engine handles remaining placement


def heuristic_purchase(state: GameState, player: str) -> np.ndarray:
    """Simple heuristic purchase: mostly infantry + some artillery/armor."""
    p_idx = PLAYER_INDEX[player]
    budget = int(state.pus[p_idx])
    purchase = np.zeros(NUM_UNIT_TYPES, dtype=np.int32)

    inf_idx = UNIT_TYPE_INDEX["infantry"]
    art_idx = UNIT_TYPE_INDEX["artillery"]
    arm_idx = UNIT_TYPE_INDEX["armour"]
    ftr_idx = UNIT_TYPE_INDEX["fighter"]

    # Buy 1 fighter if affordable
    if budget >= 10:
        purchase[ftr_idx] = 1
        budget -= 10

    # Buy some armor
    num_armor = min(budget // 5, 2)
    purchase[arm_idx] = num_armor
    budget -= num_armor * 5

    # Buy some artillery
    num_art = min(budget // 4, 2)
    purchase[art_idx] = num_art
    budget -= num_art * 4

    # Rest in infantry
    num_inf = budget // 3
    purchase[inf_idx] = num_inf

    return purchase
