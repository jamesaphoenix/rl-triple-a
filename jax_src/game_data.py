"""Static game data extracted from WW2v3 1942 XML.

All arrays are plain NumPy — loaded once at init, then converted to JAX
arrays as constants. These never change during training.
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from src.map_parser import load_default_map

# ── Constants ────────────────────────────────────────────────
NUM_UNIT_TYPES = 13   # infantry..factory
NUM_PLAYERS = 7
MAX_UNITS_PER_SLOT = 30  # cap per (territory, player, unit_type)

PLAYERS = ["Japanese", "Russians", "Germans", "British", "Italians", "Chinese", "Americans"]
PLAYER_IDX = {p: i for i, p in enumerate(PLAYERS)}
AXIS_MASK = np.array([1, 0, 1, 0, 1, 0, 0], dtype=np.int32)   # 1 = Axis
ALLIED_MASK = 1 - AXIS_MASK

UNIT_NAMES = [
    "infantry", "artillery", "armour", "fighter", "bomber",
    "transport", "submarine", "destroyer", "cruiser", "carrier",
    "battleship", "aaGun", "factory",
]
UNIT_IDX = {n: i for i, n in enumerate(UNIT_NAMES)}

# Unit stats: attack, defense, movement, cost
UNIT_ATTACK  = np.array([1, 2, 3, 3, 4, 0, 2, 2, 3, 1, 4, 0, 0], dtype=np.int32)
UNIT_DEFENSE = np.array([2, 2, 3, 4, 1, 0, 1, 2, 3, 2, 4, 0, 0], dtype=np.int32)
UNIT_COST    = np.array([3, 4, 5, 10, 12, 7, 6, 8, 12, 14, 20, 6, 15], dtype=np.int32)
UNIT_MOVE    = np.array([1, 1, 2, 4, 6, 2, 2, 2, 2, 2, 2, 1, 0], dtype=np.int32)
UNIT_IS_LAND = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1], dtype=np.int32)
UNIT_IS_SEA  = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0], dtype=np.int32)
UNIT_IS_AIR  = np.array([0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int32)
# Non-combat units (factory, aaGun) — excluded from battle rolls
UNIT_IS_COMBAT = np.array([1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0], dtype=np.int32)
# Purchasable (not factory/aa for standard action space, but include all)
UNIT_PURCHASABLE = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.int32)


def build_map_arrays():
    """Load the WW2v3 1942 map and return NumPy arrays for JAX.

    Returns dict with:
        num_territories: int
        territory_names: list[str]
        adjacency: (T, T) bool — True if territories are connected
        is_water: (T,) bool
        is_impassable: (T,) bool
        production: (T,) int — PU production value
        is_victory_city: (T,) bool
        is_capital: (T,) int — player index if capital, -1 otherwise
        initial_owner: (T,) int — player index, -1 if unowned
        initial_units: (T, P, U) int — starting unit counts
        initial_pus: (P,) int
        has_factory: (T,) bool — initial factory presence
    """
    md = load_default_map()

    # Sort territories deterministically
    names = sorted(md.territories.keys())
    T = len(names)
    tidx = {n: i for i, n in enumerate(names)}

    adjacency = np.zeros((T, T), dtype=bool)
    is_water = np.zeros(T, dtype=bool)
    is_impassable = np.zeros(T, dtype=bool)
    production = np.zeros(T, dtype=np.int32)
    is_victory_city = np.zeros(T, dtype=bool)
    is_capital = np.full(T, -1, dtype=np.int32)
    initial_owner = np.full(T, -1, dtype=np.int32)
    initial_units = np.zeros((T, NUM_PLAYERS, NUM_UNIT_TYPES), dtype=np.int32)
    has_factory = np.zeros(T, dtype=bool)

    for name in names:
        td = md.territories[name]
        i = tidx[name]
        is_water[i] = td.is_water
        is_impassable[i] = td.is_impassable
        production[i] = td.production
        is_victory_city[i] = td.is_victory_city

        if td.capital_of and td.capital_of in PLAYER_IDX:
            is_capital[i] = PLAYER_IDX[td.capital_of]

        if td.owner and td.owner in PLAYER_IDX:
            initial_owner[i] = PLAYER_IDX[td.owner]

        for neighbor in td.neighbors:
            if neighbor in tidx:
                adjacency[i, tidx[neighbor]] = True

        for owner, units in td.units.items():
            if owner not in PLAYER_IDX:
                continue
            pi = PLAYER_IDX[owner]
            for uname, count in units.items():
                if uname in UNIT_IDX:
                    ui = UNIT_IDX[uname]
                    initial_units[i, pi, ui] = count
                    if uname == "factory":
                        has_factory[i] = True

    initial_pus = np.zeros(NUM_PLAYERS, dtype=np.int32)
    for pname, pd in md.players.items():
        if pname in PLAYER_IDX:
            initial_pus[PLAYER_IDX[pname]] = pd.starting_pus

    return {
        "num_territories": T,
        "territory_names": names,
        "adjacency": adjacency,
        "is_water": is_water,
        "is_impassable": is_impassable,
        "production": production,
        "is_victory_city": is_victory_city,
        "is_capital": is_capital,
        "initial_owner": initial_owner,
        "initial_units": initial_units,
        "initial_pus": initial_pus,
        "has_factory": has_factory,
    }


# Singleton — loaded once
_MAP_DATA = None

def get_map_data():
    global _MAP_DATA
    if _MAP_DATA is None:
        _MAP_DATA = build_map_arrays()
    return _MAP_DATA
