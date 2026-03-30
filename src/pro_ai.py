"""ProAI-inspired heuristic opponent for Axis players.

Implements territory-value-based decision making similar to TripleA's ProAI:
- Purchase: prioritizes defense of threatened territories, then offense
- Combat moves: attacks territories where TUV swing is favorable
- Non-combat: reinforces frontline and threatened capitals
- Placement: distributes to most threatened factories
"""

from __future__ import annotations
from collections import defaultdict

import numpy as np

from .game_state import (
    GameState, ALL_PLAYERS, AXIS_PLAYERS, ALLIED_PLAYERS,
    PLAYER_INDEX,
)
from .units import (
    PURCHASABLE_UNITS, UNIT_TYPE_INDEX, NUM_UNIT_TYPES,
    UnitDomain,
)
from .combat import estimate_battle_odds


def pro_purchase(state: GameState, player: str) -> np.ndarray:
    """ProAI-inspired purchase strategy.

    Priority:
    1. If capital is threatened, buy max infantry for defense
    2. Otherwise, buy a balanced mix based on income:
       - Low income (<15): infantry + artillery
       - Medium (15-30): infantry + artillery + armor + 1 fighter
       - High (>30): balanced army + navy if coastal
    """
    p_idx = PLAYER_INDEX[player]
    budget = int(state.pus[p_idx])
    purchase = np.zeros(NUM_UNIT_TYPES, dtype=np.int32)
    alliance = "Axis" if player in AXIS_PLAYERS else "Allies"
    enemy_players = ALLIED_PLAYERS if alliance == "Axis" else AXIS_PLAYERS

    inf_idx = UNIT_TYPE_INDEX["infantry"]
    art_idx = UNIT_TYPE_INDEX["artillery"]
    arm_idx = UNIT_TYPE_INDEX["armour"]
    ftr_idx = UNIT_TYPE_INDEX["fighter"]
    bmb_idx = UNIT_TYPE_INDEX["bomber"]
    sub_idx = UNIT_TYPE_INDEX["submarine"]
    dd_idx = UNIT_TYPE_INDEX["destroyer"]
    trn_idx = UNIT_TYPE_INDEX["transport"]

    # Find capital
    capital_idx = None
    for t in state.territories:
        if t.capital_of == player and t.owner == player:
            capital_idx = t.index
            break

    # Check if capital is threatened
    capital_threatened = False
    if capital_idx is not None:
        cap = state.territories[capital_idx]
        enemy_strength = 0
        for n_idx in cap.neighbor_indices:
            nt = state.territories[n_idx]
            for ep in enemy_players:
                if ep in PLAYER_INDEX:
                    ep_idx = PLAYER_INDEX[ep]
                    for u in range(NUM_UNIT_TYPES):
                        ut = PURCHASABLE_UNITS[u]
                        if not ut.is_factory and not ut.is_aa:
                            enemy_strength += nt.units[ep_idx, u] * ut.attack

        # Calculate our defense at capital
        our_defense = 0
        for u in range(NUM_UNIT_TYPES):
            ut = PURCHASABLE_UNITS[u]
            if not ut.is_factory and not ut.is_aa:
                our_defense += cap.units[p_idx, u] * ut.defense

        if enemy_strength > our_defense * 0.6:
            capital_threatened = True

    if capital_threatened:
        # Emergency defense: max infantry
        num_inf = budget // 3
        purchase[inf_idx] = num_inf
        return purchase

    # Normal purchasing
    income = state.get_player_income(player)

    if income < 15:
        # Low income: infantry + some artillery
        num_art = min(budget // 4, 1)
        purchase[art_idx] = num_art
        budget -= num_art * 4
        purchase[inf_idx] = budget // 3
    elif income <= 30:
        # Medium income: balanced ground + air
        if budget >= 10:
            purchase[ftr_idx] = 1
            budget -= 10
        num_arm = min(budget // 5, 2)
        purchase[arm_idx] = num_arm
        budget -= num_arm * 5
        num_art = min(budget // 4, 2)
        purchase[art_idx] = num_art
        budget -= num_art * 4
        purchase[inf_idx] = budget // 3
    else:
        # High income: player-specific strategies
        if player == "Germans":
            # Germany: heavy land army for Eastern Front push toward Moscow
            # 1 fighter + lots of armor + infantry screen
            if budget >= 10:
                purchase[ftr_idx] = 1
                budget -= 10
            num_arm = min(budget // 5, 4)
            purchase[arm_idx] = num_arm
            budget -= num_arm * 5
            num_art = min(budget // 4, 3)
            purchase[art_idx] = num_art
            budget -= num_art * 4
            purchase[inf_idx] = budget // 3

        elif player == "Japanese":
            # Japan: combined arms - navy + transports for island hopping + land for Asia
            if budget >= 10:
                purchase[ftr_idx] = 1
                budget -= 10
            # Transport + escort for Pacific pressure
            if budget >= 15:
                purchase[trn_idx] = 1
                budget -= 7
                purchase[dd_idx] = 1
                budget -= 8
            num_arm = min(budget // 5, 2)
            purchase[arm_idx] = num_arm
            budget -= num_arm * 5
            num_art = min(budget // 4, 2)
            purchase[art_idx] = num_art
            budget -= num_art * 4
            purchase[inf_idx] = budget // 3

        elif player == "Italians":
            # Italy: cheap units, maybe 1 armor for North Africa push
            num_arm = min(budget // 5, 1)
            purchase[arm_idx] = num_arm
            budget -= num_arm * 5
            purchase[inf_idx] = budget // 3

        else:
            # Generic high-income: full combined arms
            if budget >= 10:
                purchase[ftr_idx] = 1
                budget -= 10
            if budget >= 12 and np.random.random() > 0.5:
                purchase[bmb_idx] = 1
                budget -= 12
            num_arm = min(budget // 5, 3)
            purchase[arm_idx] = num_arm
            budget -= num_arm * 5
            num_art = min(budget // 4, 3)
            purchase[art_idx] = num_art
            budget -= num_art * 4
            purchase[inf_idx] = budget // 3

    return purchase


def pro_combat_moves(
    state: GameState, player: str
) -> list[tuple[int, int, np.ndarray]]:
    """ProAI-inspired combat movement.

    Strategy:
    1. Identify all attackable enemy territories
    2. For each, calculate expected TUV swing using battle odds
    3. Prioritize attacks with positive TUV swing and high territory value
    4. Don't attack if it would leave our territories undefended
    """
    p_idx = PLAYER_INDEX[player]
    alliance = "Axis" if player in AXIS_PLAYERS else "Allies"
    enemy_players = ALLIED_PLAYERS if alliance == "Axis" else AXIS_PLAYERS
    friendly_players = AXIS_PLAYERS if alliance == "Axis" else ALLIED_PLAYERS

    rng = np.random.default_rng()
    moves = []
    committed_units = defaultdict(lambda: np.zeros(NUM_UNIT_TYPES, dtype=np.int32))

    # Score all potential attack targets
    targets = []
    for t in state.territories:
        if t.is_impassable or t.is_water:
            continue
        if t.owner not in enemy_players:
            continue

        # Sum defenders
        dfn = np.zeros(NUM_UNIT_TYPES, dtype=np.int32)
        for ep in enemy_players:
            if ep in PLAYER_INDEX:
                dfn += t.units[PLAYER_INDEX[ep]]

        dfn_combat = sum(dfn[i] for i in range(NUM_UNIT_TYPES)
                        if not PURCHASABLE_UNITS[i].is_factory
                        and not PURCHASABLE_UNITS[i].is_aa)

        # Gather available attackers from adjacent territories
        available_by_source = {}
        total_atk = np.zeros(NUM_UNIT_TYPES, dtype=np.int32)

        for n_idx in t.neighbor_indices:
            nt = state.territories[n_idx]
            if nt.is_impassable:
                continue
            if nt.owner not in friendly_players and not nt.is_water:
                continue

            available = nt.units[p_idx].copy() - committed_units[n_idx]
            available = np.maximum(available, 0)

            # Keep factories and AA
            factory_idx = UNIT_TYPE_INDEX["factory"]
            aa_idx = UNIT_TYPE_INDEX["aaGun"]
            available[factory_idx] = 0
            available[aa_idx] = 0

            # Keep minimum defense (1 infantry if it's a valuable territory)
            if nt.owner == player and nt.production >= 2 and not nt.is_water:
                inf_idx = UNIT_TYPE_INDEX["infantry"]
                keep = min(2, int(available[inf_idx]))
                available[inf_idx] = max(0, available[inf_idx] - keep)

            if np.sum(available) > 0:
                available_by_source[n_idx] = available
                total_atk += available

        if np.sum(total_atk) == 0:
            continue

        # Estimate battle odds
        odds = estimate_battle_odds(total_atk, dfn, num_simulations=50, rng=rng)

        # Score: win probability * territory value - expected TUV loss
        tuv_swing = odds["avg_defender_tuv_loss"] - odds["avg_attacker_tuv_loss"]
        territory_value = t.production * 2
        if t.is_victory_city:
            territory_value += 10
        if t.is_capital:
            territory_value += 20

        score = (odds["attacker_win_pct"] * territory_value + tuv_swing)

        targets.append({
            "territory": t,
            "dfn": dfn,
            "dfn_combat": dfn_combat,
            "total_atk": total_atk,
            "sources": available_by_source,
            "odds": odds,
            "score": score,
        })

    # Sort by score (best attacks first)
    targets.sort(key=lambda x: x["score"], reverse=True)

    # Execute attacks with positive expected value
    for target in targets:
        # Only attack if >65% win probability and positive TUV swing
        # (or undefended territories)
        if target["dfn_combat"] == 0:
            # Undefended: send 1 unit
            for src_idx, available in target["sources"].items():
                for u_idx in range(NUM_UNIT_TYPES):
                    ut = PURCHASABLE_UNITS[u_idx]
                    if ut.is_factory or ut.is_aa or ut.domain != UnitDomain.LAND:
                        continue
                    avail = int(available[u_idx] - committed_units[src_idx][u_idx])
                    if avail > 0:
                        units = np.zeros(NUM_UNIT_TYPES, dtype=np.int32)
                        units[u_idx] = 1
                        moves.append((src_idx, target["territory"].index, units))
                        committed_units[src_idx][u_idx] += 1
                        break
                break
            continue

        # More aggressive: attack at 50% if high-value, 55% otherwise
        min_win_pct = 0.50 if target["territory"].production >= 3 or target["territory"].is_victory_city else 0.55
        if target["odds"]["attacker_win_pct"] < min_win_pct:
            continue

        tuv_swing = target["odds"]["avg_defender_tuv_loss"] - target["odds"]["avg_attacker_tuv_loss"]
        # Accept negative TUV swing for high-value targets (capitals, VCs)
        if tuv_swing < -10 and not target["territory"].is_victory_city and target["territory"].production < 3:
            continue

        # Commit units from all sources
        for src_idx, available in target["sources"].items():
            remaining = available - committed_units[src_idx]
            remaining = np.maximum(remaining, 0)
            if np.sum(remaining) > 0:
                moves.append((src_idx, target["territory"].index, remaining))
                committed_units[src_idx] += remaining

    return moves


def pro_noncombat_moves(
    state: GameState, player: str
) -> list[tuple[int, int, np.ndarray]]:
    """ProAI-inspired non-combat movement.

    Strategy:
    1. Move units toward the front line (territories adjacent to enemies)
    2. Reinforce threatened territories
    3. Stack units at strategic chokepoints
    """
    p_idx = PLAYER_INDEX[player]
    alliance = "Axis" if player in AXIS_PLAYERS else "Allies"
    enemy_players = ALLIED_PLAYERS if alliance == "Axis" else AXIS_PLAYERS
    friendly_players = AXIS_PLAYERS if alliance == "Axis" else ALLIED_PLAYERS

    moves = []

    # Find frontline territories (friendly territories adjacent to enemies)
    frontline = []
    rear = []
    for t in state.territories:
        if t.is_impassable or t.is_water:
            continue
        if t.owner not in friendly_players:
            continue

        is_front = False
        for n_idx in t.neighbor_indices:
            nt = state.territories[n_idx]
            if nt.owner in enemy_players and not nt.is_impassable:
                is_front = True
                break

        if is_front:
            frontline.append(t)
        else:
            rear.append(t)

    # Find rear territories with movable units
    for t in rear:
        units = t.units[p_idx].copy()
        factory_idx = UNIT_TYPE_INDEX["factory"]
        aa_idx = UNIT_TYPE_INDEX["aaGun"]
        units[factory_idx] = 0
        units[aa_idx] = 0

        mobile_count = np.sum(units)
        if mobile_count == 0:
            continue

        # Find best frontline territory to move toward
        # (closest frontline territory via shared neighbors)
        best_neighbor = None
        best_score = -1

        for n_idx in t.neighbor_indices:
            nt = state.territories[n_idx]
            if nt.is_impassable or nt.is_water:
                continue

            # Score: prefer moving toward frontline
            score = 0
            if nt in frontline:
                score += 10 + nt.production
            else:
                # Is it closer to frontline?
                for nn_idx in nt.neighbor_indices:
                    nnt = state.territories[nn_idx]
                    if nnt in frontline:
                        score += 5
                        break

            if score > best_score:
                best_score = score
                best_neighbor = n_idx

        if best_neighbor is not None and best_score > 0:
            # Move most land units (keep 1 infantry if territory has production)
            to_move = units.copy()
            if t.production > 0:
                inf_idx = UNIT_TYPE_INDEX["infantry"]
                if to_move[inf_idx] > 0:
                    to_move[inf_idx] -= 1

            # Only move land units in non-combat
            for u_idx in range(NUM_UNIT_TYPES):
                ut = PURCHASABLE_UNITS[u_idx]
                if ut.domain != UnitDomain.LAND:
                    to_move[u_idx] = 0

            if np.sum(to_move) > 0:
                moves.append((t.index, best_neighbor, to_move))

    return moves


def pro_placement(
    state: GameState, player: str, pending_purchases: dict[int, int]
) -> list[tuple[int, np.ndarray]]:
    """ProAI-inspired unit placement.

    Strategy:
    1. Place at most threatened factory first
    2. Spread remaining across factories by need
    3. Sea units go to adjacent sea zone of most important factory
    """
    p_idx = PLAYER_INDEX[player]
    alliance = "Axis" if player in AXIS_PLAYERS else "Allies"
    enemy_players = ALLIED_PLAYERS if alliance == "Axis" else AXIS_PLAYERS
    factories = state.get_factories(player)

    if not factories:
        return []

    # Score factories by threat level
    factory_scores = []
    for f_idx in factories:
        t = state.territories[f_idx]
        threat = 0
        for n_idx in t.neighbor_indices:
            nt = state.territories[n_idx]
            for ep in enemy_players:
                if ep in PLAYER_INDEX:
                    ep_idx = PLAYER_INDEX[ep]
                    for u in range(NUM_UNIT_TYPES):
                        ut = PURCHASABLE_UNITS[u]
                        if not ut.is_factory and not ut.is_aa:
                            threat += nt.units[ep_idx, u] * ut.attack

        factory_scores.append((f_idx, threat, t.production))

    # Sort: most threatened first
    factory_scores.sort(key=lambda x: (x[1], x[2]), reverse=True)

    placements = []
    remaining = dict(pending_purchases)

    # Separate land and sea units
    land_remaining = {}
    sea_remaining = {}
    for u_idx, count in remaining.items():
        ut = PURCHASABLE_UNITS[u_idx]
        if ut.domain == UnitDomain.SEA:
            sea_remaining[u_idx] = count
        else:
            land_remaining[u_idx] = count

    # Place land units at factories
    for f_idx, threat, prod in factory_scores:
        t = state.territories[f_idx]
        capacity = t.factory_capacity if t.factory_capacity else t.production
        placed = 0

        units_array = np.zeros(NUM_UNIT_TYPES, dtype=np.int32)
        for u_idx in sorted(land_remaining.keys(),
                           key=lambda i: PURCHASABLE_UNITS[i].defense, reverse=True):
            count = land_remaining.get(u_idx, 0)
            can_place = min(count, max(0, capacity - placed))
            if can_place > 0:
                units_array[u_idx] = can_place
                land_remaining[u_idx] -= can_place
                if land_remaining[u_idx] == 0:
                    del land_remaining[u_idx]
                placed += can_place

        if np.sum(units_array) > 0:
            placements.append((f_idx, units_array))

    # Place sea units
    if sea_remaining:
        for f_idx, _, _ in factory_scores:
            t = state.territories[f_idx]
            for n_idx in t.neighbor_indices:
                nt = state.territories[n_idx]
                if nt.is_water:
                    units_array = np.zeros(NUM_UNIT_TYPES, dtype=np.int32)
                    for u_idx, count in list(sea_remaining.items()):
                        units_array[u_idx] = count
                        del sea_remaining[u_idx]
                    if np.sum(units_array) > 0:
                        placements.append((n_idx, units_array))
                    break
            if not sea_remaining:
                break

    return placements
