"""JAX game engine for WW2v3 1942.

Everything is a pure function operating on JAX arrays.
No Python loops in the hot path — fully JIT-compilable and vmap-able.

State is a flat dict of arrays (a pytree), enabling:
  - jax.jit: compile the entire game step
  - jax.vmap: vectorize across 100s of parallel games
  - jax.grad: differentiate through the game (future)
"""

from __future__ import annotations
from functools import partial

import jax
import jax.numpy as jnp
from jax import random

from .game_data import (
    NUM_UNIT_TYPES, NUM_PLAYERS, MAX_UNITS_PER_SLOT,
    UNIT_ATTACK, UNIT_DEFENSE, UNIT_COST, UNIT_IS_COMBAT,
    UNIT_IS_LAND, UNIT_IS_SEA, AXIS_MASK, ALLIED_MASK,
    UNIT_IDX, get_map_data,
)


# ── State ─────────────────────────────────────────────────────

def make_initial_state(rng_key):
    """Create initial game state from map data."""
    md = get_map_data()
    T = md["num_territories"]

    return {
        "units": jnp.array(md["initial_units"]),           # (T, P, U)
        "owner": jnp.array(md["initial_owner"]),            # (T,)
        "pus": jnp.array(md["initial_pus"]),                # (P,)
        "round": jnp.int32(1),
        "current_player": jnp.int32(0),                     # index into PLAYERS
        "phase": jnp.int32(0),                               # 0=purchase,1=combat,2=noncombat,3=place
        "done": jnp.bool_(False),
        "winner": jnp.int32(-1),                             # -1=none, 0=Axis, 1=Allies
        "pending_purchase": jnp.zeros(NUM_UNIT_TYPES, dtype=jnp.int32),
        "rng": rng_key,
    }


# ── Static data (converted to JAX once) ──────────────────────

def get_jax_statics():
    """Convert map data to JAX arrays. Call once, reuse everywhere."""
    md = get_map_data()
    return {
        "adjacency": jnp.array(md["adjacency"]),
        "is_water": jnp.array(md["is_water"]),
        "is_impassable": jnp.array(md["is_impassable"]),
        "production": jnp.array(md["production"]),
        "is_victory_city": jnp.array(md["is_victory_city"]),
        "is_capital": jnp.array(md["is_capital"]),
        "has_factory_init": jnp.array(md["has_factory"]),
        "num_territories": md["num_territories"],
        "unit_attack": jnp.array(UNIT_ATTACK),
        "unit_defense": jnp.array(UNIT_DEFENSE),
        "unit_cost": jnp.array(UNIT_COST),
        "unit_is_combat": jnp.array(UNIT_IS_COMBAT),
        "unit_is_land": jnp.array(UNIT_IS_LAND),
        "unit_is_sea": jnp.array(UNIT_IS_SEA),
        "axis_mask": jnp.array(AXIS_MASK),
        "allied_mask": jnp.array(ALLIED_MASK),
    }


# ── Combat ────────────────────────────────────────────────────

def resolve_combat_jax(atk_units, dfn_units, rng_key, statics, max_rounds=12):
    """Resolve one battle. All JAX, no Python control flow.

    Args:
        atk_units: (U,) attacker unit counts
        dfn_units: (U,) defender unit counts
        rng_key: JAX PRNG key
        statics: static game data
        max_rounds: max combat rounds

    Returns:
        atk_remaining: (U,)
        dfn_remaining: (U,)
        atk_wins: bool
    """
    unit_atk = statics["unit_attack"]
    unit_def = statics["unit_defense"]
    is_combat = statics["unit_is_combat"]

    def combat_round(carry, _):
        atk, dfn, key = carry
        key, k1, k2 = random.split(key, 3)

        # Roll attack dice: each unit rolls d6, hits if roll <= attack value
        # Total hits = sum of (count * probability)
        # For speed, use expected hits + binomial noise
        atk_combat = atk * is_combat
        dfn_combat = dfn * is_combat

        # Attack rolls
        atk_hit_prob = jnp.clip(unit_atk / 6.0, 0.0, 1.0)
        atk_expected = (atk_combat * atk_hit_prob).sum()
        atk_hits = jnp.round(atk_expected + random.normal(k1) * jnp.sqrt(
            (atk_combat * atk_hit_prob * (1 - atk_hit_prob)).sum()
        )).astype(jnp.int32)
        atk_hits = jnp.clip(atk_hits, 0, dfn_combat.sum())

        # Defense rolls
        dfn_hit_prob = jnp.clip(unit_def / 6.0, 0.0, 1.0)
        dfn_expected = (dfn_combat * dfn_hit_prob).sum()
        dfn_hits = jnp.round(dfn_expected + random.normal(k2) * jnp.sqrt(
            (dfn_combat * dfn_hit_prob * (1 - dfn_hit_prob)).sum()
        )).astype(jnp.int32)
        dfn_hits = jnp.clip(dfn_hits, 0, atk_combat.sum())

        # Apply casualties (cheapest units die first)
        # Sort by cost — fixed order: infantry(0), artillery(1), armour(2), ...
        # Remove from defender using atk_hits
        dfn_new = _apply_casualties(dfn, atk_hits, is_combat)
        atk_new = _apply_casualties(atk, dfn_hits, is_combat)

        return (atk_new, dfn_new, key), None

    (atk_final, dfn_final, _), _ = jax.lax.scan(
        combat_round, (atk_units, dfn_units, rng_key), None, length=max_rounds
    )

    atk_alive = (atk_final * is_combat).sum() > 0
    dfn_alive = (dfn_final * is_combat).sum() > 0
    atk_wins = atk_alive & (~dfn_alive)

    return atk_final, dfn_final, atk_wins


def _apply_casualties(units, hits, is_combat):
    """Remove `hits` casualties from cheapest combat units first.

    Unit order is already sorted by cost (infantry=0 is cheapest).
    """
    # Process each unit type from cheapest (idx 0) to most expensive
    def remove_one_type(carry, ui):
        remaining_hits, current_units = carry
        can_remove = jnp.where(is_combat[ui], jnp.minimum(current_units[ui], remaining_hits), 0)
        new_units = current_units.at[ui].add(-can_remove)
        remaining_hits = remaining_hits - can_remove
        return (remaining_hits, new_units), None

    (_, result), _ = jax.lax.scan(
        remove_one_type,
        (hits, units),
        jnp.arange(NUM_UNIT_TYPES)
    )
    return result


# ── Turn Execution ────────────────────────────────────────────

def execute_purchase(state, purchase_action, statics):
    """Execute purchase phase.

    Args:
        state: game state dict
        purchase_action: (U,) int array of unit counts to buy
        statics: static data

    Returns:
        new_state
    """
    p = state["current_player"]
    budget = state["pus"][p]
    cost = statics["unit_cost"]

    # Clamp to affordable
    purchase = jnp.clip(purchase_action, 0, 20).astype(jnp.int32)
    total_cost = (purchase * cost).sum()

    # Scale down if over budget
    scale = jnp.where(total_cost > 0, jnp.minimum(1.0, budget / total_cost), 1.0)
    purchase = jnp.floor(purchase * scale).astype(jnp.int32)
    total_cost = (purchase * cost).sum()

    new_pus = state["pus"].at[p].add(-total_cost)
    return {**state, "pus": new_pus, "pending_purchase": purchase}


def execute_combat_and_battles(state, attack_scores, statics):
    """Execute combat movement + battle resolution in one step.

    Args:
        state: game state
        attack_scores: (T,) float — score for each territory (higher = attack)
        statics: static data

    Returns:
        new_state, tuv_swing
    """
    p = state["current_player"]
    T = statics["num_territories"]
    adj = statics["adjacency"]
    is_imp = statics["is_impassable"]
    is_water = statics["is_water"]
    axis_mask = statics["axis_mask"]

    is_axis = axis_mask[p]
    friendly_mask = jnp.where(is_axis, axis_mask, 1 - axis_mask)  # (P,)
    enemy_mask = 1 - friendly_mask

    units = state["units"]  # (T, P, U)
    owner = state["owner"]  # (T,)
    rng = state["rng"]

    # For each territory, check if it's an enemy territory worth attacking
    # Enemy territory: owned by enemy player, not impassable, attack_score > threshold
    owner_is_enemy = jnp.array([
        jnp.where(owner[t] >= 0, enemy_mask[owner[t]], 0)
        for t in range(T)
    ])  # (T,)
    should_attack = (attack_scores > 0.3) & (owner_is_enemy > 0) & (~is_imp) & (~is_water)

    def resolve_territory(carry, t_idx):
        current_units, current_owner, key = carry

        key, subkey = random.split(key)
        do_attack = should_attack[t_idx]

        # Sum enemy defenders in this territory
        dfn = jnp.zeros(NUM_UNIT_TYPES, dtype=jnp.int32)
        for pi in range(NUM_PLAYERS):
            dfn = dfn + current_units[t_idx, pi] * enemy_mask[pi]

        # Sum friendly attackers from adjacent territories
        atk = jnp.zeros(NUM_UNIT_TYPES, dtype=jnp.int32)
        neighbors = adj[t_idx]  # (T,) bool
        for pi in range(NUM_PLAYERS):
            # Sum units of friendly player pi across all adjacent territories
            friendly_units = current_units[:, pi, :] * neighbors[:, None]  # (T, U)
            atk = atk + friendly_units.sum(axis=0) * friendly_mask[pi]

        # Only keep combat units for attack
        atk = atk * statics["unit_is_combat"]

        has_battle = do_attack & (atk.sum() > 0) & (dfn.sum() > 0)

        atk_remaining, dfn_remaining, atk_wins = resolve_combat_jax(
            atk, dfn, subkey, statics
        )

        # If attacker wins and we should attack: conquer territory
        conquer = has_battle & atk_wins
        new_owner = jnp.where(conquer, p, current_owner[t_idx])
        current_owner = current_owner.at[t_idx].set(new_owner)

        # If attacker wins: replace enemy units with attacker remaining
        # (simplified: all go to current player)
        def apply_battle_result(cu):
            # Zero out enemy units in conquered territory
            for pi in range(NUM_PLAYERS):
                cu = jnp.where(
                    conquer & (enemy_mask[pi] > 0),
                    cu.at[t_idx, pi].set(jnp.zeros(NUM_UNIT_TYPES, dtype=jnp.int32)),
                    cu
                )
            # Set attacker's remaining units
            cu = jnp.where(
                conquer,
                cu.at[t_idx, p].set(atk_remaining),
                cu
            )
            return cu

        current_units = apply_battle_result(current_units)

        return (current_units, current_owner, key), None

    (final_units, final_owner, final_key), _ = jax.lax.scan(
        resolve_territory,
        (units, owner, rng),
        jnp.arange(T),
    )

    new_state = {
        **state,
        "units": final_units,
        "owner": final_owner,
        "rng": final_key,
    }

    # TUV swing approximation
    old_tuv = (units[:, :, :] * statics["unit_cost"][None, None, :]).sum()
    new_tuv = (final_units[:, :, :] * statics["unit_cost"][None, None, :]).sum()
    tuv_swing = old_tuv - new_tuv  # positive = units were destroyed

    return new_state, tuv_swing


def execute_placement(state, statics):
    """Place pending purchases at factories. Automatic — first factory with capacity."""
    p = state["current_player"]
    T = statics["num_territories"]
    production = statics["production"]
    purchase = state["pending_purchase"]

    units = state["units"]
    owner = state["owner"]
    factory_idx = UNIT_IDX["factory"]

    # Find territories with factories owned by current player
    has_factory = units[:, p, factory_idx] > 0  # (T,)
    is_mine = owner == p  # (T,)
    valid_factory = has_factory & is_mine  # (T,)

    # Place at first valid factory
    # Simple: put everything at the factory with highest production
    factory_scores = valid_factory.astype(jnp.float32) * production
    best_factory = jnp.argmax(factory_scores)

    # Add land units to factory territory
    land_mask = jnp.array(UNIT_IS_LAND, dtype=jnp.int32)
    land_purchase = purchase * land_mask
    new_units = units.at[best_factory, p].add(land_purchase)

    # Add sea units to adjacent sea zone
    sea_mask = jnp.array(UNIT_IS_SEA, dtype=jnp.int32)
    sea_purchase = purchase * sea_mask
    adj_sea = statics["adjacency"][best_factory] & statics["is_water"]
    sea_zone = jnp.argmax(adj_sea.astype(jnp.int32))
    has_sea = adj_sea.any()
    new_units = jnp.where(has_sea, new_units.at[sea_zone, p].add(sea_purchase), new_units)

    return {
        **state,
        "units": new_units,
        "pending_purchase": jnp.zeros(NUM_UNIT_TYPES, dtype=jnp.int32),
    }


def end_turn(state, statics):
    """Collect income, advance player, check victory."""
    p = state["current_player"]
    T = statics["num_territories"]
    production = statics["production"]
    owner = state["owner"]
    is_vc = statics["is_victory_city"]
    axis_mask = statics["axis_mask"]

    # Collect income: sum production of owned territories
    my_territories = (owner == p).astype(jnp.int32)
    income = (my_territories * production).sum()
    new_pus = state["pus"].at[p].add(income)

    # Advance to next player
    next_player = (p + 1) % NUM_PLAYERS
    new_round = jnp.where(next_player == 0, state["round"] + 1, state["round"])

    # Check victory: count victory cities per alliance
    axis_vc = jnp.int32(0)
    allied_vc = jnp.int32(0)
    for t in range(T):
        t_owner = owner[t]
        is_vc_t = is_vc[t]
        is_axis_owner = jnp.where(t_owner >= 0, axis_mask[t_owner], 0)
        axis_vc = axis_vc + (is_vc_t & (is_axis_owner > 0)).astype(jnp.int32)
        allied_vc = allied_vc + (is_vc_t & (is_axis_owner == 0) & (t_owner >= 0)).astype(jnp.int32)

    # Game ends if one side has 15+ VCs or after 15 rounds
    axis_wins = axis_vc >= 15
    allied_wins = allied_vc >= 15
    time_up = new_round > 15
    game_over = axis_wins | allied_wins | time_up

    winner = jnp.where(axis_wins, 0, jnp.where(allied_wins, 1,
             jnp.where(time_up, jnp.where(axis_vc > allied_vc, 0, 1), -1)))

    return {
        **state,
        "pus": new_pus,
        "current_player": next_player,
        "round": new_round,
        "done": game_over,
        "winner": winner,
    }


# ── Full Step ─────────────────────────────────────────────────

def game_step(state, action, statics):
    """Execute one full turn: purchase → combat → placement → end.

    Args:
        state: game state
        action: dict with "purchase" (U,) and "attack_scores" (T,)
        statics: static data

    Returns:
        new_state, reward
    """
    # Purchase
    state = execute_purchase(state, action["purchase"], statics)

    # Combat + battles
    state, tuv_swing = execute_combat_and_battles(state, action["attack_scores"], statics)

    # Placement
    state = execute_placement(state, statics)

    # End turn
    pre_vc_allies = _count_allied_vc(state, statics)
    state = end_turn(state, statics)
    post_vc_allies = _count_allied_vc(state, statics)

    # Reward (from Allied perspective)
    is_axis = statics["axis_mask"][state["current_player"] - 1]  # prev player
    sign = jnp.where(is_axis, -1.0, 1.0)
    reward = sign * (tuv_swing * 0.001 + (post_vc_allies - pre_vc_allies) * 1.0)

    # Terminal reward
    reward = jnp.where(
        state["done"] & (state["winner"] == 1),  # Allies win
        reward + 100.0,
        jnp.where(state["done"] & (state["winner"] == 0), reward - 100.0, reward)
    )

    return state, reward


def _count_allied_vc(state, statics):
    owner = state["owner"]
    is_vc = statics["is_victory_city"]
    axis_mask = statics["axis_mask"]
    allied_vc = jnp.int32(0)
    T = statics["num_territories"]
    for t in range(T):
        t_owner = owner[t]
        is_allied = jnp.where(t_owner >= 0, 1 - axis_mask[t_owner], 0)
        allied_vc = allied_vc + (is_vc[t] & (is_allied > 0)).astype(jnp.int32)
    return allied_vc


# ── Observation ───────────────────────────────────────────────

def state_to_obs(state, statics):
    """Convert state to flat observation vector.

    Features per territory (T entries):
        - owner one-hot (P)
        - units per player per type (P * U)
        - production (1)
        - is_water (1)

    Global:
        - PUs per player (P)
        - round (1)
        - current player one-hot (P)

    Total: T * (P + P*U + 2) + 2*P + 1
    """
    T = statics["num_territories"]
    units = state["units"]  # (T, P, U)
    owner = state["owner"]  # (T,)

    # Owner one-hot: (T, P)
    owner_onehot = jax.nn.one_hot(jnp.clip(owner, 0, NUM_PLAYERS - 1), NUM_PLAYERS)
    owner_onehot = owner_onehot * (owner >= 0)[:, None]  # zero out unowned

    # Units normalized
    units_flat = units.reshape(T, -1).astype(jnp.float32) / 10.0  # (T, P*U)

    # Per-territory scalars
    prod = statics["production"][:, None].astype(jnp.float32) / 12.0
    water = statics["is_water"][:, None].astype(jnp.float32)

    # Concat per-territory features
    terr_features = jnp.concatenate([owner_onehot, units_flat, prod, water], axis=1)  # (T, F)
    terr_flat = terr_features.reshape(-1)

    # Global features
    pus = state["pus"].astype(jnp.float32) / 50.0
    rnd = jnp.array([state["round"].astype(jnp.float32) / 20.0])
    player_onehot = jax.nn.one_hot(state["current_player"], NUM_PLAYERS)

    obs = jnp.concatenate([terr_flat, pus, rnd, player_onehot])
    return obs
