"""Parse WW2v3 1942 XML map definition into Python data structures."""

from __future__ import annotations
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class TerritoryData:
    name: str
    is_water: bool = False
    production: int = 0
    is_impassable: bool = False
    owner: Optional[str] = None
    original_owner: Optional[str] = None
    is_capital: bool = False
    capital_of: Optional[str] = None
    is_victory_city: bool = False
    unit_production: Optional[int] = None  # factory capacity override
    units: dict[str, dict[str, int]] = field(default_factory=dict)  # owner -> {unit_type: count}
    neighbors: list[str] = field(default_factory=list)


@dataclass
class PlayerData:
    name: str
    alliance: str
    starting_pus: int = 0


@dataclass
class NationalObjective:
    player: str
    value: int = 0
    territories: list[str] = field(default_factory=list)
    count: int = 0
    description: str = ""


@dataclass
class MapData:
    territories: dict[str, TerritoryData] = field(default_factory=dict)
    players: dict[str, PlayerData] = field(default_factory=dict)
    turn_order: list[str] = field(default_factory=list)
    alliances: dict[str, str] = field(default_factory=dict)  # player -> alliance
    national_objectives: list[NationalObjective] = field(default_factory=list)
    canals: list[dict] = field(default_factory=list)


def parse_map(xml_path: str | Path) -> MapData:
    """Parse a TripleA game XML file into MapData."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    data = MapData()

    # Parse territories
    map_elem = root.find("map")
    if map_elem is None:
        raise ValueError("No <map> element found")

    for t in map_elem.findall("territory"):
        name = t.get("name", "")
        is_water = t.get("water", "false").lower() == "true"
        data.territories[name] = TerritoryData(name=name, is_water=is_water)

    # Parse connections
    for conn in map_elem.findall("connection"):
        t1 = conn.get("t1", "")
        t2 = conn.get("t2", "")
        if t1 in data.territories and t2 in data.territories:
            data.territories[t1].neighbors.append(t2)
            data.territories[t2].neighbors.append(t1)

    # Parse players
    player_list = root.find("playerList")
    if player_list is not None:
        for p in player_list.findall("player"):
            name = p.get("name", "")
            data.players[name] = PlayerData(name=name, alliance="")

        for a in player_list.findall("alliance"):
            player = a.get("player", "")
            alliance = a.get("alliance", "")
            if player in data.players:
                data.players[player].alliance = alliance
                data.alliances[player] = alliance

        data.turn_order = [p.get("name", "") for p in player_list.findall("player")]

    # Parse territory attachments (production values, capitals, etc.)
    attachment_list = root.find("attachmentList")
    if attachment_list is not None:
        for att in attachment_list.findall("attachment"):
            attach_to = att.get("attachTo", "")
            java_class = att.get("javaClass", "")

            if "TerritoryAttachment" in java_class and attach_to in data.territories:
                t = data.territories[attach_to]
                for opt in att.findall("option"):
                    name = opt.get("name", "")
                    value = opt.get("value", "")
                    if name == "production":
                        t.production = int(value)
                    elif name == "isImpassable" and value.lower() == "true":
                        t.is_impassable = True
                    elif name == "capital":
                        t.is_capital = True
                        t.capital_of = value
                    elif name == "victoryCity":
                        t.is_victory_city = int(value) > 0
                    elif name == "originalOwner":
                        t.original_owner = value
                    elif name == "unitProduction":
                        t.unit_production = int(value)

            # Parse national objectives
            if "RulesAttachment" in java_class and "objective" in att.get("name", ""):
                obj = NationalObjective(player=attach_to)
                for opt in att.findall("option"):
                    name = opt.get("name", "")
                    value = opt.get("value", "")
                    if name == "objectiveValue":
                        obj.value = int(value)
                    elif name == "alliedOwnershipTerritories":
                        obj.territories = value.split(":")
                    elif name == "count":
                        obj.count = int(value)
                data.national_objectives.append(obj)

            # Parse canals
            if "CanalAttachment" in java_class:
                canal = {"sea_zone": attach_to}
                for opt in att.findall("option"):
                    name = opt.get("name", "")
                    value = opt.get("value", "")
                    if name == "canalName":
                        canal["name"] = value
                    elif name == "landTerritories":
                        canal["land_territories"] = value.split(":")
                data.canals.append(canal)

    # Parse initial ownership
    init = root.find("initialize")
    if init is not None:
        owner_init = init.find("ownerInitialize")
        if owner_init is not None:
            for to in owner_init.findall("territoryOwner"):
                territory = to.get("territory", "")
                owner = to.get("owner", "")
                if territory in data.territories:
                    data.territories[territory].owner = owner

        # Parse initial unit placements
        unit_init = init.find("unitInitialize")
        if unit_init is not None:
            for up in unit_init.findall("unitPlacement"):
                unit_type = up.get("unitType", "")
                territory = up.get("territory", "")
                quantity = int(up.get("quantity", "0"))
                owner = up.get("owner", "")
                if territory in data.territories:
                    t = data.territories[territory]
                    if owner not in t.units:
                        t.units[owner] = {}
                    t.units[owner][unit_type] = t.units[owner].get(unit_type, 0) + quantity

        # Parse initial resources
        res_init = init.find("resourceInitialize")
        if res_init is not None:
            for rg in res_init.findall("resourceGiven"):
                player = rg.get("player", "")
                resource = rg.get("resource", "")
                quantity = int(rg.get("quantity", "0"))
                if player in data.players and resource == "PUs":
                    data.players[player].starting_pus = quantity

    return data


# Default map path
DEFAULT_MAP = Path(__file__).parent.parent / "triplea" / "game-app" / "game-core" / \
    "src" / "testFixtures" / "resources" / "ww2v3_1942_test.xml"


def load_default_map() -> MapData:
    """Load the WW2v3 1942 map."""
    return parse_map(DEFAULT_MAP)
