"""Export map data as numpy arrays for the Rust engine."""

import numpy as np
from .map_parser import load_default_map

PLAYERS = ["Japanese", "Russians", "Germans", "British", "Italians", "Chinese", "Americans"]
PLAYER_IDX = {p: i for i, p in enumerate(PLAYERS)}
UNIT_NAMES = ["infantry", "artillery", "armour", "fighter", "bomber",
              "transport", "submarine", "destroyer", "cruiser", "carrier",
              "battleship", "aaGun", "factory"]
UNIT_IDX = {n: i for i, n in enumerate(UNIT_NAMES)}
NUM_PLAYERS = 7
NUM_UNIT_TYPES = 13


def export_map_arrays() -> dict:
    md = load_default_map()
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
    initial_pus = np.zeros(NUM_PLAYERS, dtype=np.int32)

    # Chinese-allowed territories (Bug #14)
    CHINESE_TERRITORY_NAMES = {
        "Chinghai", "Ningxia", "Sikang", "Yunnan", "Hupeh",
        "Fukien", "Suiyuan", "Manchuria", "Kiangsu", "Kwangtung",
    }
    chinese_territories = np.zeros(T, dtype=bool)
    for name_c in CHINESE_TERRITORY_NAMES:
        if name_c in tidx:
            chinese_territories[tidx[name_c]] = True

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
                    initial_units[i, pi, UNIT_IDX[uname]] = count

    for pname, pd in md.players.items():
        if pname in PLAYER_IDX:
            initial_pus[PLAYER_IDX[pname]] = pd.starting_pus

    # National Objectives — resolved to territory indices
    # Format: (player_name, value, [territory_names], count, [enemy_sea_zone_names])
    # Format: (player, value, territories, count, enemy_sea_zones, allied_exclusion)
    national_objectives_raw = [
        # Germans
        ("Germans", 5, ["France", "Northwestern Europe", "Germany", "Poland",
                        "Czechoslovakia Hungary", "Bulgaria Romania"], 6, [], False),
        ("Germans", 5, ["Baltic States", "East Poland", "Belorussia",
                        "Eastern Ukraine", "Ukraine"], 3, [], False),
        ("Germans", 5, ["Karelia S.S.R.", "Caucasus"], 1, [], False),
        # Russians — FIX #18: first NO has allied_exclusion=True
        ("Russians", 5, ["Archangel"], 1, [], True),
        ("Russians", 10, ["Norway", "Finland", "Poland", "Bulgaria Romania",
                          "Czechoslovakia Hungary", "Balkans"], 3, [], False),
        # Japanese
        ("Japanese", 5, ["Manchuria", "Kiangsu", "French Indo-China Thailand"], 3, [], False),
        ("Japanese", 5, ["Kwangtung", "East Indies", "Borneo", "Philippine Islands",
                         "New Guinea", "Solomon Islands"], 4, [], False),
        ("Japanese", 5, ["Hawaiian Islands", "Australia", "India"], 1, [], False),
        # British
        ("British", 5, ["Caroline Islands", "French Indo-China Thailand", "Formosa",
                        "Iwo Jima", "Japan", "Okinawa"], 1, [], False),
        ("British", 5, ["Eastern Canada", "Western Canada", "Gibraltar", "Egypt",
                        "Australia", "Union of South Africa"], 6, [], False),
        ("British", 5, ["France", "Balkans"], 1, [], False),
        # Italians
        ("Italians", 5, ["Egypt", "Trans-Jordan", "France", "Gibraltar"], 3, [], False),
        ("Italians", 5, ["Italy", "Balkans", "Morocco Algeria", "Libya"], 4,
         ["13 Sea Zone", "14 Sea Zone", "15 Sea Zone"], False),
        # Americans
        ("Americans", 5, ["France"], 1, [], False),
        ("Americans", 5, ["Philippine Islands"], 1, [], False),
        ("Americans", 5, ["Western United States", "Eastern United States",
                          "Central United States"], 3, [], False),
        ("Americans", 5, ["Midway", "Wake Island", "Hawaiian Islands",
                          "Solomon Islands"], 3, [], False),
    ]

    national_objectives = []
    for player_name, value, terr_names, count, sea_names, allied_excl in national_objectives_raw:
        player_idx = PLAYER_IDX[player_name]
        terr_indices = np.array([tidx[n] for n in terr_names if n in tidx], dtype=np.int32)
        sea_indices = np.array([tidx[n] for n in sea_names if n in tidx], dtype=np.int32)
        national_objectives.append({
            "player": player_idx,
            "value": value,
            "territories": terr_indices,
            "count": count,
            "enemy_sea_zones": sea_indices,
            "allied_exclusion": allied_excl,
        })

    return {
        "adjacency": adjacency,
        "is_water": is_water,
        "is_impassable": is_impassable,
        "production": production,
        "is_victory_city": is_victory_city,
        "is_capital": is_capital,
        "chinese_territories": chinese_territories,
        "initial_units": initial_units,
        "initial_owner": initial_owner,
        "initial_pus": initial_pus,
        "territory_names": names,
        "national_objectives": national_objectives,
    }
