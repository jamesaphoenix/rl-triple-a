"""Core game state for WW2v3 1942."""

from __future__ import annotations
import copy
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .map_parser import MapData, load_default_map
from .units import (
    UNIT_TYPES, PURCHASABLE_UNITS, UNIT_TYPE_INDEX, NUM_UNIT_TYPES,
    UnitDomain,
)

# Player constants
AXIS_PLAYERS = ["Germans", "Japanese", "Italians"]
ALLIED_PLAYERS = ["Russians", "British", "Chinese", "Americans"]
ALL_PLAYERS = ["Japanese", "Russians", "Germans", "British", "Italians", "Chinese", "Americans"]
PLAYER_INDEX = {p: i for i, p in enumerate(ALL_PLAYERS)}
NUM_PLAYERS = len(ALL_PLAYERS)


@dataclass
class Territory:
    name: str
    index: int
    is_water: bool
    production: int
    is_impassable: bool
    owner: Optional[str]
    original_owner: Optional[str]
    is_capital: bool
    capital_of: Optional[str]
    is_victory_city: bool
    factory_capacity: Optional[int]
    neighbor_indices: list[int] = field(default_factory=list)
    # units[player_idx][unit_type_idx] = count
    units: np.ndarray = field(default_factory=lambda: np.zeros((NUM_PLAYERS, NUM_UNIT_TYPES), dtype=np.int32))
    has_factory: bool = False


class GameState:
    """Full game state for WW2v3 1942."""

    def __init__(self, map_data: Optional[MapData] = None):
        if map_data is None:
            map_data = load_default_map()
        self.map_data = map_data

        # Build territory list (sorted for determinism)
        terr_names = sorted(map_data.territories.keys())
        self.territory_names = terr_names
        self.territory_index = {name: i for i, name in enumerate(terr_names)}
        self.num_territories = len(terr_names)

        # Build adjacency
        self.adjacency = np.zeros((self.num_territories, self.num_territories), dtype=bool)
        self.territories: list[Territory] = []

        for name in terr_names:
            td = map_data.territories[name]
            idx = self.territory_index[name]
            t = Territory(
                name=name,
                index=idx,
                is_water=td.is_water,
                production=td.production,
                is_impassable=td.is_impassable,
                owner=td.owner,
                original_owner=td.original_owner,
                is_capital=td.is_capital,
                capital_of=td.capital_of,
                is_victory_city=td.is_victory_city,
                factory_capacity=td.unit_production,
            )
            self.territories.append(t)

        # Set up adjacency and neighbor lists
        for name in terr_names:
            td = map_data.territories[name]
            idx = self.territory_index[name]
            for neighbor_name in td.neighbors:
                if neighbor_name in self.territory_index:
                    n_idx = self.territory_index[neighbor_name]
                    self.adjacency[idx, n_idx] = True
                    self.territories[idx].neighbor_indices.append(n_idx)

        # Place initial units
        for name in terr_names:
            td = map_data.territories[name]
            idx = self.territory_index[name]
            t = self.territories[idx]
            for owner, units in td.units.items():
                if owner not in PLAYER_INDEX:
                    continue
                p_idx = PLAYER_INDEX[owner]
                for unit_name, count in units.items():
                    if unit_name in UNIT_TYPE_INDEX:
                        u_idx = UNIT_TYPE_INDEX[unit_name]
                        t.units[p_idx, u_idx] = count
                        if unit_name == "factory":
                            t.has_factory = True

        # Player resources
        self.pus = np.zeros(NUM_PLAYERS, dtype=np.int32)
        for player_name, pd in map_data.players.items():
            if player_name in PLAYER_INDEX:
                self.pus[PLAYER_INDEX[player_name]] = pd.starting_pus

        # Game progression
        self.round = 1
        self.current_player_idx = 0  # index into ALL_PLAYERS
        self.phase = "purchase"  # purchase, combat_move, battle, noncombat_move, place, end_turn
        self.game_over = False
        self.winner: Optional[str] = None  # "Axis" or "Allies"

        # Precompute land territory indices (non-water, non-impassable)
        self.land_indices = [t.index for t in self.territories
                            if not t.is_water and not t.is_impassable]
        self.sea_indices = [t.index for t in self.territories if t.is_water]

        # Victory cities
        self.victory_cities = [t.index for t in self.territories if t.is_victory_city]

    @property
    def current_player(self) -> str:
        return ALL_PLAYERS[self.current_player_idx]

    @property
    def current_alliance(self) -> str:
        return "Axis" if self.current_player in AXIS_PLAYERS else "Allies"

    def get_player_income(self, player: str) -> int:
        """Calculate total PU income for a player."""
        p_idx = PLAYER_INDEX[player]
        income = 0
        for t in self.territories:
            if t.owner == player and not t.is_water:
                income += t.production
        return income

    def get_player_territories(self, player: str) -> list[int]:
        """Get indices of territories owned by player."""
        return [t.index for t in self.territories if t.owner == player]

    def get_factories(self, player: str) -> list[int]:
        """Get indices of territories where player has factories."""
        p_idx = PLAYER_INDEX[player]
        factory_idx = UNIT_TYPE_INDEX["factory"]
        result = []
        for t in self.territories:
            if t.owner == player and t.units[p_idx, factory_idx] > 0:
                result.append(t.index)
        return result

    def get_units_in_territory(self, terr_idx: int, player: str) -> dict[str, int]:
        """Get unit counts for a player in a territory."""
        p_idx = PLAYER_INDEX[player]
        t = self.territories[terr_idx]
        result = {}
        for ut_name, ut_idx in UNIT_TYPE_INDEX.items():
            count = t.units[p_idx, ut_idx]
            if count > 0:
                result[ut_name] = int(count)
        return result

    def count_victory_cities(self, alliance: str) -> int:
        """Count victory cities controlled by an alliance."""
        if alliance == "Axis":
            players = AXIS_PLAYERS
        else:
            players = ALLIED_PLAYERS
        count = 0
        for idx in self.victory_cities:
            if self.territories[idx].owner in players:
                count += 1
        return count

    def check_victory(self) -> Optional[str]:
        """Check if either side has won. Returns 'Axis', 'Allies', or None."""
        axis_vc = self.count_victory_cities("Axis")
        allied_vc = self.count_victory_cities("Allies")
        total_vc = len(self.victory_cities)

        # Projection of Power: control 15+ victory cities at end of a round
        if axis_vc >= 15 or axis_vc > total_vc * 0.7:
            return "Axis"
        if allied_vc >= 15 or allied_vc > total_vc * 0.7:
            return "Allies"

        # Capital capture check
        axis_capitals_held = 0
        allied_capitals_held = 0
        for t in self.territories:
            if t.is_capital and t.capital_of:
                if t.capital_of in AXIS_PLAYERS and t.owner in AXIS_PLAYERS:
                    axis_capitals_held += 1
                elif t.capital_of in ALLIED_PLAYERS and t.owner in ALLIED_PLAYERS:
                    allied_capitals_held += 1

        # If all enemy capitals captured
        if axis_capitals_held == 0:
            return "Allies"
        if allied_capitals_held == 0:
            return "Axis"

        return None

    def to_observation(self, perspective_player: str) -> np.ndarray:
        """Convert game state to a flat observation vector for RL.

        Features per territory:
        - is_water (1)
        - production value (1)
        - owner one-hot (NUM_PLAYERS)
        - units per player per type (NUM_PLAYERS * NUM_UNIT_TYPES)
        - has_factory (1)
        - is_friendly (1)
        - is_enemy (1)

        Plus global features:
        - PUs per player (NUM_PLAYERS)
        - round number (1)
        - current player one-hot (NUM_PLAYERS)
        """
        perspective_alliance = "Axis" if perspective_player in AXIS_PLAYERS else "Allies"
        friendly_players = AXIS_PLAYERS if perspective_alliance == "Axis" else ALLIED_PLAYERS

        per_territory = 3 + NUM_PLAYERS + NUM_PLAYERS * NUM_UNIT_TYPES + 1 + 2
        global_features = NUM_PLAYERS + 1 + NUM_PLAYERS

        obs = np.zeros(self.num_territories * per_territory + global_features, dtype=np.float32)

        for i, t in enumerate(self.territories):
            if t.is_impassable:
                continue
            offset = i * per_territory
            obs[offset] = 1.0 if t.is_water else 0.0
            obs[offset + 1] = t.production / 12.0  # normalize
            obs[offset + 2] = 1.0 if t.has_factory else 0.0

            # Owner one-hot
            if t.owner and t.owner in PLAYER_INDEX:
                obs[offset + 3 + PLAYER_INDEX[t.owner]] = 1.0

            # Units
            unit_offset = offset + 3 + NUM_PLAYERS
            for p_idx in range(NUM_PLAYERS):
                for u_idx in range(NUM_UNIT_TYPES):
                    obs[unit_offset + p_idx * NUM_UNIT_TYPES + u_idx] = \
                        t.units[p_idx, u_idx] / 10.0  # normalize

            # Friendly/enemy flags
            flag_offset = unit_offset + NUM_PLAYERS * NUM_UNIT_TYPES
            obs[flag_offset] = 1.0 if t.owner in friendly_players else 0.0
            obs[flag_offset + 1] = 1.0 if (t.owner and t.owner not in friendly_players
                                            and t.owner is not None) else 0.0

        # Global features
        g_offset = self.num_territories * per_territory
        for p_idx in range(NUM_PLAYERS):
            obs[g_offset + p_idx] = self.pus[p_idx] / 50.0
        obs[g_offset + NUM_PLAYERS] = self.round / 20.0
        obs[g_offset + NUM_PLAYERS + 1 + self.current_player_idx] = 1.0

        return obs

    def copy(self) -> GameState:
        """Deep copy the game state."""
        new = GameState.__new__(GameState)
        new.map_data = self.map_data
        new.territory_names = self.territory_names
        new.territory_index = self.territory_index
        new.num_territories = self.num_territories
        new.adjacency = self.adjacency.copy()
        new.territories = []
        for t in self.territories:
            nt = Territory(
                name=t.name, index=t.index, is_water=t.is_water,
                production=t.production, is_impassable=t.is_impassable,
                owner=t.owner, original_owner=t.original_owner,
                is_capital=t.is_capital, capital_of=t.capital_of,
                is_victory_city=t.is_victory_city,
                factory_capacity=t.factory_capacity,
                neighbor_indices=t.neighbor_indices.copy(),
                units=t.units.copy(),
                has_factory=t.has_factory,
            )
            new.territories.append(nt)
        new.pus = self.pus.copy()
        new.round = self.round
        new.current_player_idx = self.current_player_idx
        new.phase = self.phase
        new.game_over = self.game_over
        new.winner = self.winner
        new.land_indices = self.land_indices
        new.sea_indices = self.sea_indices
        new.victory_cities = self.victory_cities
        return new

    @property
    def observation_size(self) -> int:
        per_territory = 3 + NUM_PLAYERS + NUM_PLAYERS * NUM_UNIT_TYPES + 1 + 2
        global_features = NUM_PLAYERS + 1 + NUM_PLAYERS
        return self.num_territories * per_territory + global_features
