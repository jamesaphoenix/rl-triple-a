#!/usr/bin/env python3
"""Extract game state from TripleA .tsvg save files.

.tsvg files are GZIP-compressed Java serialized objects. We can't easily
deserialize them in Python, but we CAN:

1. Decompress the GZIP layer
2. Scan the binary stream for readable string data (territory names,
   unit types, player names) to reconstruct approximate game state
3. Compare sequential saves to infer moves made between phases

This gives us enough to learn your dad's Axis playstyle.
"""

from __future__ import annotations
import gzip
import struct
import re
import json
import sys
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field


# Java serialization constants
TC_OBJECT = 0x73
TC_STRING = 0x74
TC_LONG_STRING = 0x7C
TC_REFERENCE = 0x71
TC_ARRAY = 0x75
TC_BLOCKDATA = 0x77
TC_ENDBLOCKDATA = 0x78


def extract_strings_from_save(filepath: Path) -> list[str]:
    """Extract all readable strings from a .tsvg save file."""
    with gzip.open(filepath, 'rb') as f:
        data = f.read()

    strings = []
    i = 0
    while i < len(data) - 2:
        # Look for Java serialization short string marker
        if data[i] == TC_STRING:
            length = struct.unpack('>H', data[i+1:i+3])[0]
            if length < 2000 and i + 3 + length <= len(data):
                try:
                    s = data[i+3:i+3+length].decode('utf-8', errors='ignore')
                    if s and len(s) > 1:
                        strings.append(s)
                except:
                    pass
            i += 3 + length
        else:
            i += 1

    return strings


def extract_game_state_from_strings(strings: list[str]) -> dict:
    """Parse extracted strings to reconstruct approximate game state."""
    # Known territory names from WW2v3 1942
    known_territories = {
        "Afghanistan", "Alaska", "Anglo-Egypt Sudan", "Angola", "Argentina Chile",
        "Archangel", "Australia", "Balkans", "Baltic States", "Belgian Congo",
        "Belorussia", "Borneo", "Brazil", "Burma", "Buryatia S.S.R.",
        "Caroline Islands", "Caucasus", "Central United States", "Chinghai",
        "Czechoslovakia Hungary", "East Indies", "Eastern Canada", "East Poland",
        "Eastern Ukraine", "Eastern United States", "Egypt", "Eire",
        "Evenki National Okrug", "France", "French Equatorial Africa",
        "French Madagascar", "French West Africa", "Finland", "Formosa",
        "Fukien", "Germany", "Gibraltar", "Greenland", "Hawaiian Islands",
        "Himalaya", "Hupeh", "Iceland", "India", "French Indo-China Thailand",
        "Italian Africa", "Italy", "Iwo Jima", "Japan", "Karelia S.S.R.",
        "Kazakh S.S.R.", "Kiangsu", "Kwangtung", "Libya", "Manchuria",
        "Mexico", "Midway", "Mongolia", "Morocco Algeria", "Mozambique",
        "New Guinea", "New Zealand", "Ningxia", "Northwestern Europe",
        "Norway", "Novosibirsk", "Okinawa", "Panama", "Persia",
        "Peruvian Central", "Philippine Islands", "Rhodesia", "Poland",
        "Bulgaria Romania", "Russia", "Sahara", "Saudi Arabia", "Sikang",
        "Solomon Islands", "Soviet Far East", "Spain", "Stanovoj Chrebet",
        "Suiyuan", "Sweden", "Switzerland", "Trans-Jordan", "Turkey",
        "Ukraine", "Union of South Africa", "United Kingdom", "Urals",
        "Northern South America", "Wake Island", "West Indies",
        "Western Canada", "Western United States", "Yakut S.S.R.", "Yunnan",
    }
    # Add sea zones
    for i in range(1, 66):
        known_territories.add(f"{i} Sea Zone")

    known_players = {"Germans", "Russians", "Japanese", "British", "Italians",
                     "Chinese", "Americans"}
    known_units = {"infantry", "artillery", "armour", "fighter", "bomber",
                   "transport", "battleship", "carrier", "submarine", "factory",
                   "aaGun", "destroyer", "cruiser"}

    # Find territories, players, and units mentioned
    found_territories = []
    found_players = []
    found_units = []

    for s in strings:
        if s in known_territories:
            found_territories.append(s)
        elif s in known_players:
            found_players.append(s)
        elif s in known_units:
            found_units.append(s)

    return {
        "territories_mentioned": len(set(found_territories)),
        "unique_territories": list(set(found_territories))[:20],
        "players_mentioned": list(set(found_players)),
        "units_mentioned": list(set(found_units)),
        "total_strings": len(strings),
    }


def analyze_save_sequence(save_dir: Path) -> dict:
    """Analyze a sequence of autosaves to infer game progression.

    The autosave filenames tell us the game phase:
    - autosaveAfterGermanCombatMove.tsvg -> after German combat moves
    - autosaveAfterBattle.tsvg -> after battle resolution
    - autosave_round_even.tsvg / autosave_round_odd.tsvg -> round snapshots
    """
    saves = sorted(save_dir.glob("*.tsvg"))

    # Define the turn sequence ordering
    phase_order = [
        "autosaveAfterJapaneseCombatMove",
        "autosaveAfterJapaneseNonCombatMove",
        "autosaveAfterRussianCombatMove",
        "autosaveAfterRussianNonCombatMove",
        "autosaveAfterGermanCombatMove",
        "autosaveAfterGermanNonCombatMove",
        "autosaveAfterBritishCombatMove",
        "autosaveAfterBritishNonCombatMove",
        "autosaveAfterItalianCombatMove",
        "autosaveAfterItalianNonCombatMove",
        "autosaveAfterAmericanCombatMove",
        "autosaveAfterAmericanNonCombatMove",
        "autosaveAfterChineseCombatMove",
        "autosaveAfterChineseNonCombatMove",
    ]

    results = {}
    for save_path in saves:
        name = save_path.stem
        print(f"Extracting: {name} ({save_path.stat().st_size // 1024} KB)")

        strings = extract_strings_from_save(save_path)
        state = extract_game_state_from_strings(strings)

        # Detect which player's turn and what phase
        player = None
        phase = None
        for p in ["Japanese", "Russian", "German", "British", "Italian",
                   "American", "Chinese"]:
            if p.lower() in name.lower():
                player = p
                break

        if "CombatMove" in name:
            phase = "combat_move"
        elif "NonCombatMove" in name:
            phase = "noncombat_move"
        elif "Battle" in name:
            phase = "battle"
        elif "EndTurn" in name:
            phase = "end_turn"
        elif "round" in name:
            phase = "round_snapshot"

        results[name] = {
            "file": str(save_path),
            "size_kb": save_path.stat().st_size // 1024,
            "player": player,
            "phase": phase,
            **state,
        }

    return results


def extract_binary_game_data(filepath: Path) -> dict:
    """More detailed extraction: look for integer sequences that represent
    unit counts and resource values in the Java serialized data."""
    with gzip.open(filepath, 'rb') as f:
        data = f.read()

    strings = extract_strings_from_save(filepath)

    # Look for patterns: territory name followed by player name followed by
    # unit type followed by integer (count)
    # In Java serialization, integers are written as 4 big-endian bytes

    known_players = {"Germans", "Russians", "Japanese", "British", "Italians",
                     "Chinese", "Americans"}

    # Find PU values - look for "PUs" string near integer values
    pu_info = {}
    for i, s in enumerate(strings):
        if s == "PUs":
            # Look at nearby strings for player name and find associated values
            for j in range(max(0, i-5), min(len(strings), i+5)):
                if strings[j] in known_players:
                    pu_info[strings[j]] = {"found_near_PUs": True}

    # Find territory ownership patterns
    territory_owners = {}
    known_territories = set()
    for i, s in enumerate(strings):
        if "Sea Zone" in s or len(s) > 2:
            # Check if next string is a player name (ownership)
            if i + 1 < len(strings) and strings[i+1] in known_players:
                territory_owners[s] = strings[i+1]

    return {
        "raw_string_count": len(strings),
        "pu_mentions": pu_info,
        "territory_owners_detected": len(territory_owners),
        "sample_ownership": dict(list(territory_owners.items())[:20]),
    }


def main():
    save_dir = Path.home() / "triplea" / "savedGames" / "autoSave"

    if not save_dir.exists():
        print(f"Save directory not found: {save_dir}")
        sys.exit(1)

    saves = list(save_dir.glob("*.tsvg"))
    print(f"Found {len(saves)} save files in {save_dir}\n")

    # Analyze the save sequence
    results = analyze_save_sequence(save_dir)

    # Save results
    output_path = Path(__file__).parent.parent / "data"
    output_path.mkdir(exist_ok=True)

    with open(output_path / "save_analysis.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nAnalysis saved to {output_path / 'save_analysis.json'}")

    # Detailed extraction of a key save (after German combat move - dad's Axis play)
    german_save = save_dir / "autosaveAfterGermanCombatMove.tsvg"
    if german_save.exists():
        print(f"\n=== Detailed extraction: German Combat Move ===")
        detail = extract_binary_game_data(german_save)
        print(f"  Strings found: {detail['raw_string_count']}")
        print(f"  Territory owners detected: {detail['territory_owners_detected']}")
        print(f"  Sample ownership: {json.dumps(detail['sample_ownership'], indent=4)}")

        with open(output_path / "german_combat_detail.json", "w") as f:
            json.dump(detail, f, indent=2)

    # Also extract Japanese moves (Axis)
    japanese_save = save_dir / "autosaveAfterJapaneseCombatMove.tsvg"
    if japanese_save.exists():
        print(f"\n=== Detailed extraction: Japanese Combat Move ===")
        detail = extract_binary_game_data(japanese_save)
        print(f"  Strings found: {detail['raw_string_count']}")
        print(f"  Territory owners detected: {detail['territory_owners_detected']}")


if __name__ == "__main__":
    main()
