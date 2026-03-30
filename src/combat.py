"""Combat resolution for WW2v3 1942.

Uses Monte Carlo dice simulation matching TripleA's combat rules:
- Each unit rolls one die (1d6), hits if roll <= attack/defense value
- Artillery gives +1 attack to one paired infantry
- Submarines get first-strike (fire before general combat, if no destroyer present)
- Battleships take 2 hits
- Transports have 0 combat value and die last
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from .units import (
    UNIT_TYPES, PURCHASABLE_UNITS, UNIT_TYPE_INDEX, NUM_UNIT_TYPES,
    UnitDomain, INFANTRY, ARTILLERY, SUBMARINE, DESTROYER, BATTLESHIP, TRANSPORT,
)


@dataclass
class BattleResult:
    attacker_wins: bool
    defender_wins: bool
    draw: bool  # both eliminated
    attacker_remaining: np.ndarray  # unit counts by type
    defender_remaining: np.ndarray
    attacker_losses_tuv: float  # total unit value lost
    defender_losses_tuv: float


def resolve_combat(
    attackers: np.ndarray,  # shape (NUM_UNIT_TYPES,) - unit counts
    defenders: np.ndarray,
    is_amphibious: bool = False,
    rng: np.random.Generator | None = None,
    max_rounds: int = 50,
) -> BattleResult:
    """Resolve a single battle using dice simulation.

    Args:
        attackers: Array of attacker unit counts indexed by UNIT_TYPE_INDEX
        defenders: Array of defender unit counts indexed by UNIT_TYPE_INDEX
        is_amphibious: Whether this is an amphibious assault
        rng: Random number generator
        max_rounds: Maximum combat rounds before draw
    """
    if rng is None:
        rng = np.random.default_rng()

    atk = attackers.copy().astype(np.float64)
    dfn = defenders.copy().astype(np.float64)

    # Track battleship damage (extra hit point)
    bb_idx = UNIT_TYPE_INDEX["battleship"]
    atk_bb_damaged = 0
    dfn_bb_damaged = 0

    initial_atk_tuv = _calc_tuv(atk)
    initial_dfn_tuv = _calc_tuv(dfn)

    sub_idx = UNIT_TYPE_INDEX["submarine"]
    dd_idx = UNIT_TYPE_INDEX["destroyer"]
    transport_idx = UNIT_TYPE_INDEX["transport"]

    for _ in range(max_rounds):
        if np.sum(atk) == 0 or np.sum(dfn) == 0:
            break

        # Submarine first-strike phase
        atk_sub_hits = 0
        dfn_sub_hits = 0

        # Attacker subs fire first if defender has no destroyer
        if atk[sub_idx] > 0 and dfn[dd_idx] == 0:
            atk_sub_hits = _roll_hits(int(atk[sub_idx]),
                                       PURCHASABLE_UNITS[sub_idx].attack, rng)
            dfn = _apply_casualties(dfn, atk_sub_hits, exclude_subs=True)

        # Defender subs fire first if attacker has no destroyer
        if dfn[sub_idx] > 0 and atk[dd_idx] == 0:
            dfn_sub_hits = _roll_hits(int(dfn[sub_idx]),
                                       PURCHASABLE_UNITS[sub_idx].defense, rng)
            atk = _apply_casualties(atk, dfn_sub_hits, exclude_subs=True)

        if np.sum(atk) == 0 or np.sum(dfn) == 0:
            break

        # Main combat phase
        atk_hits = _roll_attack(atk, rng)
        dfn_hits = _roll_defense(dfn, rng)

        # Subtract hits from sub first-strike (already counted)
        # Subs that already fired in first-strike don't fire again
        if atk[sub_idx] > 0 and dfn[dd_idx] == 0:
            atk_hits -= atk_sub_hits
        if dfn[sub_idx] > 0 and atk[dd_idx] == 0:
            dfn_hits -= dfn_sub_hits

        atk_hits = max(0, atk_hits)
        dfn_hits = max(0, dfn_hits)

        # Apply battleship extra hits
        while dfn_hits > 0 and atk_bb_damaged < atk[bb_idx]:
            atk_bb_damaged += 1
            dfn_hits -= 1
        while atk_hits > 0 and dfn_bb_damaged < dfn[bb_idx]:
            dfn_bb_damaged += 1
            atk_hits -= 1

        # Apply remaining casualties
        dfn = _apply_casualties(dfn, atk_hits)
        atk = _apply_casualties(atk, dfn_hits)

        # Cap damaged battleships to remaining
        atk_bb_damaged = min(atk_bb_damaged, int(atk[bb_idx]))
        dfn_bb_damaged = min(dfn_bb_damaged, int(dfn[bb_idx]))

    atk_remaining = np.maximum(atk, 0).astype(np.int32)
    dfn_remaining = np.maximum(dfn, 0).astype(np.int32)

    atk_alive = np.sum(atk_remaining) > 0
    dfn_alive = np.sum(dfn_remaining) > 0

    return BattleResult(
        attacker_wins=atk_alive and not dfn_alive,
        defender_wins=dfn_alive and not atk_alive,
        draw=not atk_alive and not dfn_alive,
        attacker_remaining=atk_remaining,
        defender_remaining=dfn_remaining,
        attacker_losses_tuv=initial_atk_tuv - _calc_tuv(atk_remaining),
        defender_losses_tuv=initial_dfn_tuv - _calc_tuv(dfn_remaining),
    )


def _roll_hits(num_units: int, hit_value: int, rng: np.random.Generator) -> int:
    """Roll dice for units, count hits (roll <= hit_value on d6)."""
    if num_units <= 0 or hit_value <= 0:
        return 0
    rolls = rng.integers(1, 7, size=num_units)
    return int(np.sum(rolls <= hit_value))


def _roll_attack(units: np.ndarray, rng: np.random.Generator) -> int:
    """Roll attack dice for all units, with artillery support."""
    total_hits = 0
    inf_idx = UNIT_TYPE_INDEX["infantry"]
    art_idx = UNIT_TYPE_INDEX["artillery"]

    for i, ut in enumerate(PURCHASABLE_UNITS):
        count = int(units[i])
        if count <= 0 or ut.domain == UnitDomain.LAND and ut.is_factory:
            continue
        if ut.is_aa:
            continue

        attack_val = ut.attack

        # Infantry get +1 from artillery support
        if i == inf_idx and units[art_idx] > 0:
            supported = min(count, int(units[art_idx]))
            total_hits += _roll_hits(supported, attack_val + 1, rng)
            total_hits += _roll_hits(count - supported, attack_val, rng)
        else:
            total_hits += _roll_hits(count, attack_val, rng)

    return total_hits


def _roll_defense(units: np.ndarray, rng: np.random.Generator) -> int:
    """Roll defense dice for all units."""
    total_hits = 0
    for i, ut in enumerate(PURCHASABLE_UNITS):
        count = int(units[i])
        if count <= 0 or ut.is_factory or ut.is_aa:
            continue
        total_hits += _roll_hits(count, ut.defense, rng)
    return total_hits


def _apply_casualties(
    units: np.ndarray,
    hits: int,
    exclude_subs: bool = False,
) -> np.ndarray:
    """Remove casualties from unit array, cheapest units first.

    Transports die last (they have 0 combat value).
    """
    if hits <= 0:
        return units

    units = units.copy()
    transport_idx = UNIT_TYPE_INDEX["transport"]
    sub_idx = UNIT_TYPE_INDEX["submarine"]

    # Sort unit types by cost (cheapest die first), but transports last
    casualty_order = sorted(
        range(NUM_UNIT_TYPES),
        key=lambda i: (
            i == transport_idx,  # transports last
            PURCHASABLE_UNITS[i].cost,
        )
    )

    remaining_hits = hits
    for i in casualty_order:
        if remaining_hits <= 0:
            break
        if exclude_subs and i == sub_idx:
            continue
        if PURCHASABLE_UNITS[i].is_factory or PURCHASABLE_UNITS[i].is_aa:
            continue
        remove = min(int(units[i]), remaining_hits)
        units[i] -= remove
        remaining_hits -= remove

    return units


def _calc_tuv(units: np.ndarray) -> float:
    """Calculate Total Unit Value."""
    tuv = 0.0
    for i, ut in enumerate(PURCHASABLE_UNITS):
        tuv += units[i] * ut.cost
    return tuv


def estimate_battle_odds(
    attackers: np.ndarray,
    defenders: np.ndarray,
    num_simulations: int = 100,
    rng: np.random.Generator | None = None,
) -> dict:
    """Run Monte Carlo simulations to estimate win probability.

    Returns dict with keys: attacker_win_pct, defender_win_pct, draw_pct,
    avg_attacker_tuv_loss, avg_defender_tuv_loss
    """
    if rng is None:
        rng = np.random.default_rng()

    atk_wins = 0
    dfn_wins = 0
    draws = 0
    total_atk_tuv_loss = 0.0
    total_dfn_tuv_loss = 0.0

    for _ in range(num_simulations):
        result = resolve_combat(attackers, defenders, rng=rng)
        if result.attacker_wins:
            atk_wins += 1
        elif result.defender_wins:
            dfn_wins += 1
        else:
            draws += 1
        total_atk_tuv_loss += result.attacker_losses_tuv
        total_dfn_tuv_loss += result.defender_losses_tuv

    return {
        "attacker_win_pct": atk_wins / num_simulations,
        "defender_win_pct": dfn_wins / num_simulations,
        "draw_pct": draws / num_simulations,
        "avg_attacker_tuv_loss": total_atk_tuv_loss / num_simulations,
        "avg_defender_tuv_loss": total_dfn_tuv_loss / num_simulations,
    }
