"""
RL Action Server for TripleA Bot

FastAPI server that accepts game state from the RLBot Java class,
runs inference through the trained ActorCriticV3 model via the Rust
game engine, and returns concrete purchase/move/place actions as JSON.

Endpoints:
    POST /api/action  -  Accepts game state, returns phase-appropriate actions.
    GET  /health      -  Health check.

Usage:
    conda run -n rl-triplea python bot/action_server.py

    Or with a specific model:
    RL_MODEL_PATH=checkpoints_phase2/selfplay_final.pt python bot/action_server.py

    Or with a specific side:
    RL_SIDE=axis python bot/action_server.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

# Ensure project root is on sys.path for src.* imports
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.game_data_export import export_map_arrays
from src.network import ActorCriticV3
from src.units import UNIT_TYPES
from triplea_engine import TripleAEngine

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PLAYERS = ["Japanese", "Russians", "Germans", "British", "Italians", "Chinese", "Americans"]
PLAYER_IDX = {p: i for i, p in enumerate(PLAYERS)}
UNIT_NAMES = [
    "infantry", "artillery", "armour", "fighter", "bomber",
    "transport", "submarine", "destroyer", "cruiser", "carrier",
    "battleship", "aaGun", "factory",
]
UNIT_IDX = {n: i for i, n in enumerate(UNIT_NAMES)}
NUM_PLAYERS = 7
NUM_UNIT_TYPES = 13
AXIS_PLAYERS = {"Japanese", "Germans", "Italians"}

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("rl_action_server")

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="TripleA RL Action Server",
    description="Receives game state from the RLBot Java client and returns actions.",
    version="1.0.0",
)

# ---------------------------------------------------------------------------
# Pydantic schemas for request / response
# ---------------------------------------------------------------------------


class UnitInfo(BaseModel):
    type: str
    owner: str
    hits: int = 0


class TerritoryInfo(BaseModel):
    name: str
    owner: str
    isWater: bool = False
    production: int = 0
    units: list[UnitInfo] = []
    neighbors: list[str] = []


class ProductionRuleResult(BaseModel):
    name: str
    quantity: int = 1


class ProductionRuleInfo(BaseModel):
    name: str
    cost: int = 0
    results: list[ProductionRuleResult] = []


class UnitToPlace(BaseModel):
    type: str
    owner: str


class GameState(BaseModel):
    phase: str
    currentPlayer: str
    round: int = 1
    pus: int = 0
    territories: list[TerritoryInfo] = []
    productionRules: list[ProductionRuleInfo] = []
    unitsToPlace: list[UnitToPlace] = []


# ---------------------------------------------------------------------------
# Engine + Model singleton
# ---------------------------------------------------------------------------


class RLInferenceEngine:
    """Holds the Rust engine, neural net, and map data for inference."""

    def __init__(self, model_path: str, side: str = "allied"):
        self.arrays = export_map_arrays()
        self.territory_names: list[str] = self.arrays["territory_names"]
        self.tidx: dict[str, int] = {n: i for i, n in enumerate(self.territory_names)}
        self.num_t: int = len(self.territory_names)
        self.adjacency: np.ndarray = self.arrays["adjacency"]

        # Create Rust engine for observation encoding
        self.engine = TripleAEngine(
            self.arrays["adjacency"],
            self.arrays["is_water"],
            self.arrays["is_impassable"],
            self.arrays["production"],
            self.arrays["is_victory_city"],
            self.arrays["is_capital"],
            self.arrays["chinese_territories"],
            self.arrays["initial_units"],
            self.arrays["initial_owner"],
            self.arrays["initial_pus"],
            seed=0,
        )
        for no in self.arrays["national_objectives"]:
            self.engine.add_national_objective(
                no["player"], no["value"], no["territories"],
                no["count"], no["enemy_sea_zones"],
                no.get("allied_exclusion", False),
                no.get("direct_ownership", False),
            )
        for canal in self.arrays.get("canals", []):
            self.engine.add_canal(
                canal["sea_zone_a"], canal["sea_zone_b"], canal["land_territories"],
            )

        obs_size = self.engine.get_obs_size()
        action_dim = NUM_UNIT_TYPES + self.num_t + self.num_t  # 337
        self.action_dim = action_dim

        # Build adjacency matrix for V3
        adj_matrix = torch.from_numpy(self.adjacency.astype(np.float32))

        # Load model
        self.device = torch.device("cpu")
        self.model = ActorCriticV3(obs_size, action_dim, adj_matrix=adj_matrix).to(self.device)
        self.side = side

        if model_path and Path(model_path).exists():
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            state_key = "axis" if side == "axis" else "allied"
            if state_key in checkpoint:
                self.model.load_state_dict(checkpoint[state_key])
                logger.info("Loaded %s model from %s", state_key, model_path)
            elif "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
                logger.info("Loaded model from %s", model_path)
            else:
                self.model.load_state_dict(checkpoint)
                logger.info("Loaded model from %s", model_path)

            if "iteration" in checkpoint:
                logger.info(
                    "Checkpoint: iter=%d, games=%s, allied_wins=%s, axis_wins=%s",
                    checkpoint.get("iteration", "?"),
                    checkpoint.get("total_games", "?"),
                    checkpoint.get("allied_wins", "?"),
                    checkpoint.get("axis_wins", "?"),
                )
        else:
            logger.warning("No model at '%s' — using random weights", model_path)

        self.model.eval()
        logger.info(
            "Engine ready: %d territories, obs_size=%d, action_dim=%d, side=%s",
            self.num_t, obs_size, action_dim, side,
        )

    def game_state_to_observation(self, state: GameState) -> np.ndarray:
        """Convert a GameState from RLBot into a Rust engine observation vector."""
        T = self.num_t
        tidx = self.tidx

        owners = np.full(T, -1, dtype=np.int32)
        units = np.zeros((T, NUM_PLAYERS, NUM_UNIT_TYPES), dtype=np.int32)
        pus = np.zeros(NUM_PLAYERS, dtype=np.int32)

        # Build a per-territory unit map for later use
        for terr in state.territories:
            if terr.name not in tidx:
                continue
            i = tidx[terr.name]
            if terr.owner in PLAYER_IDX:
                owners[i] = PLAYER_IDX[terr.owner]
            for unit in terr.units:
                if unit.owner not in PLAYER_IDX:
                    continue
                pi = PLAYER_IDX[unit.owner]
                ui = UNIT_IDX.get(unit.type, -1)
                if ui >= 0:
                    units[i, pi, ui] += 1

        # Set PUs for the current player
        p_idx = PLAYER_IDX.get(state.currentPlayer, 0)
        pus[p_idx] = state.pus

        obs = np.array(
            self.engine.load_state(owners, units, pus, state.round, p_idx),
            dtype=np.float32,
        )
        return obs

    def infer(self, state: GameState) -> np.ndarray:
        """Run neural net inference, return 337-dim action vector in [0, 1]."""
        obs = self.game_state_to_observation(state)
        obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_mean, value = self.model.forward(obs_tensor)
        action = action_mean.squeeze(0).cpu().numpy()
        self._last_value = float(value.item())
        return action

    def decode_purchase(self, action: np.ndarray, state: GameState) -> dict[str, Any]:
        """Decode purchase scores into budget-constrained unit purchases."""
        purchase_scores = action[:NUM_UNIT_TYPES]
        budget = state.pus

        # Build cost lookup from productionRules if available
        unit_costs: dict[str, int] = {}
        for rule in state.productionRules:
            for result in rule.results:
                unit_costs[result.name.lower()] = rule.cost

        # Fall back to known costs
        for uname, ut in UNIT_TYPES.items():
            if uname.lower() not in unit_costs:
                unit_costs[uname.lower()] = ut.cost

        purchases = []
        remaining = budget

        # Sort by model score descending
        for idx in np.argsort(-purchase_scores):
            score = float(purchase_scores[idx])
            if score < 0.05 or remaining <= 0:
                break
            uname = UNIT_NAMES[idx]
            cost = unit_costs.get(uname.lower(), 99)
            if cost <= 0:
                continue
            count = min(int(score * 30), remaining // cost)
            if count > 0:
                purchases.append({"unitType": uname, "quantity": count})
                remaining -= count * cost

        return {"purchases": purchases}

    def decode_moves(self, action: np.ndarray, state: GameState) -> dict[str, Any]:
        """Decode attack/reinforce scores into concrete move orders."""
        is_combat = state.phase.lower() == "combatmove"
        player = state.currentPlayer

        if is_combat:
            scores = action[NUM_UNIT_TYPES:NUM_UNIT_TYPES + self.num_t]
        else:
            scores = action[NUM_UNIT_TYPES + self.num_t:]

        # Build territory lookup
        terr_map: dict[str, TerritoryInfo] = {t.name: t for t in state.territories}
        tidx = self.tidx
        adj = self.adjacency

        # Track which units have already been assigned to a move
        used_units: dict[str, set[int]] = {}  # territory -> set of unit indices

        moves = []
        for t_idx in np.argsort(-scores):
            score = float(scores[t_idx])
            if score < 0.3:
                break
            tname = self.territory_names[t_idx]
            if tname not in terr_map:
                continue
            target = terr_map[tname]

            if is_combat:
                # Attack: target must be enemy-owned (or neutral with units)
                if target.owner == player:
                    continue
            else:
                # Reinforce: target must be own territory
                if target.owner != player:
                    continue

            # Find our units in adjacent territories that can move here
            for n_idx in range(self.num_t):
                if not adj[t_idx, n_idx]:
                    continue
                n_name = self.territory_names[n_idx]
                if n_name not in terr_map:
                    continue
                source = terr_map[n_name]
                if source.owner != player:
                    continue

                # Get player units in source, excluding already-assigned ones
                used = used_units.get(n_name, set())
                available_units = []
                for ui, unit in enumerate(source.units):
                    if unit.owner != player:
                        continue
                    if ui in used:
                        continue
                    # Skip non-movable units
                    if unit.type.lower() in ("factory", "aagun"):
                        continue
                    available_units.append((ui, unit))

                if not available_units:
                    continue

                # For non-combat, keep 1 infantry for defense in production territories
                units_to_send = []
                keep_infantry = (
                    not is_combat
                    and source.production >= 2
                )
                infantry_kept = False

                for ui, unit in available_units:
                    if keep_infantry and unit.type.lower() == "infantry" and not infantry_kept:
                        infantry_kept = True
                        continue
                    units_to_send.append((ui, unit))

                if not units_to_send:
                    continue

                # Mark these units as used
                if n_name not in used_units:
                    used_units[n_name] = set()
                for ui, _ in units_to_send:
                    used_units[n_name].add(ui)

                unit_types = [u.type for _, u in units_to_send]
                moves.append({
                    "unitTypes": unit_types,
                    "from": n_name,
                    "to": tname,
                    "route": [n_name, tname],
                })

            if len(moves) >= 20:
                break

        return {"moves": moves}

    def decode_placement(self, action: np.ndarray, state: GameState) -> dict[str, Any]:
        """Decode placement: place units at factories, distributed by reinforce scores."""
        if not state.unitsToPlace:
            return {"placements": []}

        player = state.currentPlayer
        reinforce_scores = action[NUM_UNIT_TYPES + self.num_t:]

        # Find player's factories sorted by reinforce score
        factories = []
        for terr in state.territories:
            if terr.owner != player or terr.isWater:
                continue
            has_factory = any(
                u.type.lower() in ("factory", "industrialcomplex")
                for u in terr.units if u.owner == player
            )
            if not has_factory:
                continue
            t_idx = self.tidx.get(terr.name, -1)
            score = float(reinforce_scores[t_idx]) if t_idx >= 0 else 0.0
            factories.append((terr, score))

        factories.sort(key=lambda x: -x[1])

        if not factories:
            # Fall back to highest production owned territory
            owned = sorted(
                [t for t in state.territories if t.owner == player and not t.isWater],
                key=lambda t: -t.production,
            )
            if owned:
                factories = [(owned[0], 1.0)]

        if not factories:
            return {"placements": []}

        # Group units to place by type
        type_counts: dict[str, int] = {}
        for u in state.unitsToPlace:
            type_counts[u.type] = type_counts.get(u.type, 0) + 1

        # Place all at the top factory (simple strategy — TripleA handles overflow)
        target = factories[0][0]
        placements = [
            {"territory": target.name, "unitType": utype, "quantity": count}
            for utype, count in type_counts.items()
        ]

        return {"placements": placements}

    def get_action(self, state: GameState) -> dict[str, Any]:
        """Full inference: game state → phase-appropriate action dict."""
        action = self.infer(state)
        phase = state.phase.lower()

        if phase == "purchase":
            return self.decode_purchase(action, state)
        elif phase == "combatmove":
            return self.decode_moves(action, state)
        elif phase == "noncombatmove":
            return self.decode_moves(action, state)
        elif phase == "place":
            return self.decode_placement(action, state)
        elif phase == "tech":
            return {"skip": True}
        else:
            logger.warning("Unknown phase '%s', returning empty action.", phase)
            return {}


# ---------------------------------------------------------------------------
# Heuristic policy (used when no trained model is available)
# ---------------------------------------------------------------------------


def heuristic_purchase(state: GameState) -> dict[str, Any]:
    if not state.productionRules:
        return {"purchases": []}
    cheapest = min(state.productionRules, key=lambda r: r.cost if r.cost > 0 else 9999)
    if cheapest.cost <= 0:
        return {"purchases": []}
    quantity = state.pus // cheapest.cost
    if quantity <= 0:
        return {"purchases": []}
    unit_type = cheapest.results[0].name if cheapest.results else cheapest.name
    return {"purchases": [{"unitType": unit_type, "quantity": quantity}]}


def heuristic_move(state: GameState) -> dict[str, Any]:
    moves: list[dict[str, Any]] = []
    player = state.currentPlayer
    territory_map: dict[str, TerritoryInfo] = {t.name: t for t in state.territories}

    for terr in state.territories:
        if terr.owner != player or terr.isWater:
            continue
        my_units = [u for u in terr.units if u.owner == player]
        if not my_units:
            continue
        enemy_neighbors = [
            n for n in terr.neighbors
            if n in territory_map
            and not territory_map[n].isWater
            and territory_map[n].owner != player
            and territory_map[n].owner != "Neutral"
        ]
        if state.phase == "combatMove" and enemy_neighbors:
            target = enemy_neighbors[0]
            moves.append({
                "unitTypes": [u.type for u in my_units],
                "from": terr.name,
                "to": target,
                "route": [terr.name, target],
            })
        elif state.phase == "nonCombatMove" and not enemy_neighbors:
            for neighbor_name in terr.neighbors:
                if neighbor_name not in territory_map:
                    continue
                neighbor = territory_map[neighbor_name]
                if neighbor.owner != player or neighbor.isWater:
                    continue
                has_enemy_border = any(
                    nn in territory_map
                    and not territory_map[nn].isWater
                    and territory_map[nn].owner != player
                    and territory_map[nn].owner != "Neutral"
                    for nn in neighbor.neighbors
                )
                if has_enemy_border:
                    moves.append({
                        "unitTypes": [u.type for u in my_units],
                        "from": terr.name,
                        "to": neighbor_name,
                        "route": [terr.name, neighbor_name],
                    })
                    break
    return {"moves": moves}


def heuristic_place(state: GameState) -> dict[str, Any]:
    if not state.unitsToPlace:
        return {"placements": []}
    player = state.currentPlayer
    owned = sorted(
        [t for t in state.territories if t.owner == player and not t.isWater],
        key=lambda t: t.production, reverse=True,
    )
    if not owned:
        return {"placements": []}
    target = owned[0]
    for terr in owned:
        if any("factory" in u.type.lower() for u in terr.units if u.owner == player):
            target = terr
            break
    type_counts: dict[str, int] = {}
    for u in state.unitsToPlace:
        type_counts[u.type] = type_counts.get(u.type, 0) + 1
    return {
        "placements": [
            {"territory": target.name, "unitType": utype, "quantity": count}
            for utype, count in type_counts.items()
        ]
    }


def heuristic_policy(state: GameState) -> dict[str, Any]:
    phase = state.phase.lower()
    if phase == "purchase":
        return heuristic_purchase(state)
    elif phase in ("combatmove", "noncombatmove"):
        return heuristic_move(state)
    elif phase == "place":
        return heuristic_place(state)
    elif phase == "tech":
        return {"skip": True}
    else:
        logger.warning("Unknown phase '%s', returning empty action.", state.phase)
        return {}


# ---------------------------------------------------------------------------
# Inference engine singleton
# ---------------------------------------------------------------------------
_inference_engine: RLInferenceEngine | None = None

MODEL_PATH = os.environ.get(
    "RL_MODEL_PATH",
    str(Path(__file__).resolve().parent.parent / "checkpoints_phase2" / "selfplay_final.pt"),
)
RL_SIDE = os.environ.get("RL_SIDE", "allied")


def get_inference_engine() -> RLInferenceEngine | None:
    global _inference_engine
    if _inference_engine is not None:
        return _inference_engine

    if not Path(MODEL_PATH).exists():
        logger.warning("No model at '%s'. Using heuristic policy.", MODEL_PATH)
        return None

    try:
        _inference_engine = RLInferenceEngine(MODEL_PATH, side=RL_SIDE)
        return _inference_engine
    except Exception:
        logger.exception("Failed to load inference engine, falling back to heuristic.")
        return None


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------


@app.post("/api/action")
async def get_action(request: Request) -> dict[str, Any]:
    """Main endpoint: receives game state, returns phase-appropriate actions."""
    try:
        body = await request.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")

    try:
        state = GameState(**body)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid game state: {e}")

    logger.info(
        "Received %s phase for '%s' (round %d, %d PUs, %d territories)",
        state.phase, state.currentPlayer, state.round, state.pus,
        len(state.territories),
    )

    engine = get_inference_engine()
    if engine is not None:
        try:
            result = engine.get_action(state)
        except Exception:
            logger.exception("Model inference failed, falling back to heuristic.")
            result = heuristic_policy(state)
    else:
        result = heuristic_policy(state)

    logger.info("Returning: %s", json.dumps(result, default=str)[:500])
    return result


@app.get("/health")
async def health() -> dict[str, Any]:
    """Health check endpoint."""
    engine = get_inference_engine()
    return {
        "status": "ok",
        "model_loaded": engine is not None,
        "model_path": MODEL_PATH,
        "side": RL_SIDE,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Eagerly load the model on startup
    eng = get_inference_engine()
    if eng:
        logger.info("Model loaded successfully. Ready to serve.")
    else:
        logger.warning("No model loaded. Server will use heuristic policy.")

    uvicorn.run(
        "bot.action_server:app",
        host="0.0.0.0",
        port=8080,
        reload=False,
        log_level="info",
    )
