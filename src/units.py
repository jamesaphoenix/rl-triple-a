"""Unit type definitions for WW2v3 1942."""

from dataclasses import dataclass
from enum import Enum, auto


class UnitDomain(Enum):
    LAND = auto()
    AIR = auto()
    SEA = auto()


@dataclass(frozen=True)
class UnitType:
    name: str
    attack: int
    defense: int
    movement: int
    cost: int
    domain: UnitDomain
    hit_points: int = 1
    transport_cost: int = 0  # cost to load on a transport (0 = can't be transported)
    transport_capacity: int = 0  # how much it can carry
    can_blitz: bool = False
    can_bombard: bool = False
    is_sub: bool = False
    is_destroyer: bool = False
    is_factory: bool = False
    is_aa: bool = False
    carrier_capacity: int = 0  # how many fighters it can carry
    carrier_cost: int = 0  # cost to land on a carrier
    is_strategic_bomber: bool = False
    artillery_support: bool = False  # gives +1 attack to infantry


# Standard unit definitions from the WW2v3 1942 XML
INFANTRY = UnitType("infantry", attack=1, defense=2, movement=1, cost=3,
                     domain=UnitDomain.LAND, transport_cost=2)
ARTILLERY = UnitType("artillery", attack=2, defense=2, movement=1, cost=4,
                      domain=UnitDomain.LAND, transport_cost=3, artillery_support=True)
ARMOUR = UnitType("armour", attack=3, defense=3, movement=2, cost=5,
                   domain=UnitDomain.LAND, transport_cost=3, can_blitz=True)
FIGHTER = UnitType("fighter", attack=3, defense=4, movement=4, cost=10,
                    domain=UnitDomain.AIR, carrier_cost=1)
BOMBER = UnitType("bomber", attack=4, defense=1, movement=6, cost=12,
                   domain=UnitDomain.AIR, is_strategic_bomber=True)
TRANSPORT = UnitType("transport", attack=0, defense=0, movement=2, cost=7,
                      domain=UnitDomain.SEA, transport_capacity=5)
DESTROYER = UnitType("destroyer", attack=2, defense=2, movement=2, cost=8,
                      domain=UnitDomain.SEA, is_destroyer=True)
CRUISER = UnitType("cruiser", attack=3, defense=3, movement=2, cost=12,
                    domain=UnitDomain.SEA, can_bombard=True)
SUBMARINE = UnitType("submarine", attack=2, defense=1, movement=2, cost=6,
                      domain=UnitDomain.SEA, is_sub=True)
CARRIER = UnitType("carrier", attack=1, defense=2, movement=2, cost=14,
                    domain=UnitDomain.SEA, carrier_capacity=2)
BATTLESHIP = UnitType("battleship", attack=4, defense=4, movement=2, cost=20,
                       domain=UnitDomain.SEA, hit_points=2, can_bombard=True)
FACTORY = UnitType("factory", attack=0, defense=0, movement=0, cost=15,
                    domain=UnitDomain.LAND, is_factory=True)
AA_GUN = UnitType("aaGun", attack=0, defense=0, movement=1, cost=6,
                   domain=UnitDomain.LAND, transport_cost=3, is_aa=True)

# Lookup table
UNIT_TYPES: dict[str, UnitType] = {
    "infantry": INFANTRY,
    "artillery": ARTILLERY,
    "armour": ARMOUR,
    "fighter": FIGHTER,
    "bomber": BOMBER,
    "transport": TRANSPORT,
    "destroyer": DESTROYER,
    "cruiser": CRUISER,
    "submarine": SUBMARINE,
    "carrier": CARRIER,
    "battleship": BATTLESHIP,
    "factory": FACTORY,
    "aaGun": AA_GUN,
}

# Purchasable combat units (no factory/AA for simplicity in RL action space)
PURCHASABLE_UNITS = [INFANTRY, ARTILLERY, ARMOUR, FIGHTER, BOMBER,
                     TRANSPORT, SUBMARINE, DESTROYER, CRUISER, CARRIER, BATTLESHIP,
                     AA_GUN, FACTORY]

# Index for tensor encoding
UNIT_TYPE_INDEX = {ut.name: i for i, ut in enumerate(PURCHASABLE_UNITS)}
NUM_UNIT_TYPES = len(PURCHASABLE_UNITS)
