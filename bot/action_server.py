"""
RL Action Server for TripleA Bot

FastAPI server that accepts game state from the RLBot Java class,
runs inference through the trained RL model, and returns actions
as JSON.

Endpoints:
    POST /api/action  -  Accepts game state, returns phase-appropriate actions.

Usage:
    pip install fastapi uvicorn pydantic
    uvicorn action_server:app --host 0.0.0.0 --port 8080

    Or directly:
    python action_server.py
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

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
    version="0.1.0",
)

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
MODEL_PATH = os.environ.get("RL_MODEL_PATH", "model.pt")
_model = None


def load_model() -> Any:
    """
    Load the trained RL model from disk.

    Supports:
      - PyTorch (.pt / .pth) files via torch.load
      - Stable-Baselines3 zip files via SB3's load method
      - Falls back to a heuristic policy if no model file is found.

    Returns the loaded model or None if no model is available.
    """
    global _model
    if _model is not None:
        return _model

    model_path = Path(MODEL_PATH)
    if not model_path.exists():
        logger.warning(
            "No model file found at '%s'. Using built-in heuristic policy.", MODEL_PATH
        )
        return None

    ext = model_path.suffix.lower()
    try:
        if ext in (".pt", ".pth"):
            import torch

            _model = torch.load(str(model_path), map_location="cpu")
            _model.eval()
            logger.info("Loaded PyTorch model from %s", model_path)
        elif ext == ".zip":
            # Stable-Baselines3 convention
            from stable_baselines3 import PPO

            _model = PPO.load(str(model_path))
            logger.info("Loaded Stable-Baselines3 model from %s", model_path)
        else:
            logger.warning("Unknown model extension '%s', using heuristic.", ext)
    except Exception:
        logger.exception("Failed to load model from %s, falling back to heuristic.", model_path)
        _model = None

    return _model


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
# Heuristic policy (used when no trained model is available)
# ---------------------------------------------------------------------------


def heuristic_purchase(state: GameState) -> dict[str, Any]:
    """
    Simple heuristic: spend all PUs on infantry (cheapest common unit).
    If infantry is not available, buy the cheapest unit.
    """
    if not state.productionRules:
        return {"purchases": []}

    # Find the cheapest unit rule
    cheapest = min(state.productionRules, key=lambda r: r.cost if r.cost > 0 else 9999)
    if cheapest.cost <= 0:
        return {"purchases": []}

    quantity = state.pus // cheapest.cost
    if quantity <= 0:
        return {"purchases": []}

    # Determine the unit type name from the results
    unit_type = cheapest.results[0].name if cheapest.results else cheapest.name
    return {"purchases": [{"unitType": unit_type, "quantity": quantity}]}


def heuristic_move(state: GameState) -> dict[str, Any]:
    """
    Simple heuristic for combat/non-combat moves:
      - During combat: move land units toward enemy territories.
      - During non-combat: move units toward owned territories with lower defense.

    For simplicity, this heuristic moves idle land units toward the nearest
    enemy-adjacent territory.
    """
    moves: list[dict[str, Any]] = []
    player = state.currentPlayer

    # Build lookups
    territory_map: dict[str, TerritoryInfo] = {t.name: t for t in state.territories}

    # Identify enemy-bordering own territories as attack staging areas
    for terr in state.territories:
        if terr.owner != player:
            continue
        if terr.isWater:
            continue

        # Find player's land units in this territory
        my_units = [u for u in terr.units if u.owner == player]
        if not my_units:
            continue

        # Check if any neighbor is enemy-owned land
        enemy_neighbors = [
            n
            for n in terr.neighbors
            if n in territory_map
            and not territory_map[n].isWater
            and territory_map[n].owner != player
            and territory_map[n].owner != "Neutral"
        ]

        if state.phase == "combatMove" and enemy_neighbors:
            # Attack: move all land units to the first enemy neighbor
            target = enemy_neighbors[0]
            unit_types = [u.type for u in my_units]
            if unit_types:
                moves.append(
                    {
                        "unitTypes": unit_types,
                        "from": terr.name,
                        "to": target,
                        "route": [terr.name, target],
                    }
                )

        elif state.phase == "nonCombatMove" and not enemy_neighbors:
            # Non-combat: move units toward the nearest front line
            # Find an adjacent own territory that borders an enemy
            for neighbor_name in terr.neighbors:
                if neighbor_name not in territory_map:
                    continue
                neighbor = territory_map[neighbor_name]
                if neighbor.owner != player or neighbor.isWater:
                    continue
                # Check if this neighbor borders an enemy
                has_enemy_border = any(
                    nn in territory_map
                    and not territory_map[nn].isWater
                    and territory_map[nn].owner != player
                    and territory_map[nn].owner != "Neutral"
                    for nn in neighbor.neighbors
                )
                if has_enemy_border:
                    unit_types = [u.type for u in my_units]
                    if unit_types:
                        moves.append(
                            {
                                "unitTypes": unit_types,
                                "from": terr.name,
                                "to": neighbor_name,
                                "route": [terr.name, neighbor_name],
                            }
                        )
                    break

    return {"moves": moves}


def heuristic_place(state: GameState) -> dict[str, Any]:
    """
    Simple heuristic: place all purchased units in the highest-production
    owned territory that has a factory (i.e., can produce units).
    Falls back to capital or any owned territory.
    """
    if not state.unitsToPlace:
        return {"placements": []}

    player = state.currentPlayer

    # Find owned territories sorted by production (descending)
    owned = sorted(
        [t for t in state.territories if t.owner == player and not t.isWater],
        key=lambda t: t.production,
        reverse=True,
    )

    if not owned:
        return {"placements": []}

    # Prefer territories that have existing factory/infrastructure units
    target = owned[0]
    for terr in owned:
        has_factory = any(
            "factory" in u.type.lower() or "industrial" in u.type.lower()
            for u in terr.units
            if u.owner == player
        )
        if has_factory:
            target = terr
            break

    # Group units by type
    type_counts: dict[str, int] = {}
    for u in state.unitsToPlace:
        type_counts[u.type] = type_counts.get(u.type, 0) + 1

    placements = [
        {"territory": target.name, "unitType": utype, "quantity": count}
        for utype, count in type_counts.items()
    ]

    return {"placements": placements}


def heuristic_policy(state: GameState) -> dict[str, Any]:
    """Dispatch to the appropriate heuristic based on the current phase."""
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
# Model-based inference
# ---------------------------------------------------------------------------


def model_inference(model: Any, state: GameState) -> dict[str, Any]:
    """
    Run the trained model on the game state and return actions.

    This is a placeholder that should be adapted to the specific model
    architecture. The model is expected to accept a feature vector derived
    from the game state and return action indices that are decoded into
    purchase/move/place instructions.

    For now, this falls back to the heuristic policy.
    """
    # TODO: Implement proper feature extraction and action decoding for your
    # trained model. The structure below is a starting point.
    try:
        import numpy as np

        features = extract_features(state)
        # Example for an SB3 PPO model:
        action, _states = model.predict(features, deterministic=True)
        return decode_action(action, state)
    except Exception:
        logger.exception("Model inference failed, falling back to heuristic.")
        return heuristic_policy(state)


def extract_features(state: GameState) -> Any:
    """
    Convert the game state into a numerical feature vector for the model.

    This should be customized to match whatever observation space the model
    was trained on. A simple encoding might include:
      - Per-territory: owner one-hot, unit counts by type, production value
      - Global: current PUs, round number

    Returns a numpy array.
    """
    import numpy as np

    # Placeholder: flatten territory counts into a simple vector
    # This MUST match the observation space used during training.
    territory_features = []
    for terr in state.territories:
        # Owner encoding: 1 if owned by current player, -1 if enemy, 0 if neutral
        if terr.owner == state.currentPlayer:
            owner_val = 1.0
        elif terr.owner == "Neutral" or terr.owner == "null":
            owner_val = 0.0
        else:
            owner_val = -1.0

        # Count units by owner
        my_units = sum(1 for u in terr.units if u.owner == state.currentPlayer)
        enemy_units = sum(
            1 for u in terr.units if u.owner != state.currentPlayer and u.owner != "Neutral"
        )

        territory_features.extend(
            [owner_val, float(terr.production), float(my_units), float(enemy_units)]
        )

    global_features = [float(state.pus), float(state.round)]
    features = np.array(global_features + territory_features, dtype=np.float32)
    return features


def decode_action(action: Any, state: GameState) -> dict[str, Any]:
    """
    Decode the model's raw action output into the JSON action format
    expected by the RLBot Java class.

    This must be customized to match the model's action space.
    Falls back to heuristic if decoding fails.
    """
    # Placeholder: fall back to heuristic
    # In a real implementation, action would be an index or vector that
    # maps to specific purchases, moves, or placements.
    logger.info("decode_action called with raw action: %s (using heuristic fallback)", action)
    return heuristic_policy(state)


# ---------------------------------------------------------------------------
# API endpoint
# ---------------------------------------------------------------------------


@app.post("/api/action")
async def get_action(request: Request) -> dict[str, Any]:
    """
    Main endpoint. Receives the full game state as JSON from the RLBot
    Java class and returns the appropriate actions for the current phase.
    """
    try:
        body = await request.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")

    try:
        state = GameState(**body)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid game state: {e}")

    logger.info(
        "Received %s phase request for player '%s' (round %d, %d PUs)",
        state.phase,
        state.currentPlayer,
        state.round,
        state.pus,
    )

    model = load_model()
    if model is not None:
        result = model_inference(model, state)
    else:
        result = heuristic_policy(state)

    logger.info("Returning action: %s", json.dumps(result, default=str)[:500])
    return result


@app.get("/health")
async def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        "action_server:app",
        host="0.0.0.0",
        port=8080,
        reload=False,
        log_level="info",
    )
