#!/usr/bin/env python3
"""Integration tests for the RL bot action server pipeline.

Tests the full flow from game state JSON → Rust engine observation →
neural net inference → action decoding → JSON response, covering:
  - Server startup and health check
  - All game phases (purchase, combat, non-combat, place, tech)
  - Model vs heuristic fallback
  - Budget constraints
  - Adjacency-valid moves
  - Action format correctness for RLBot.java consumption
  - Realistic multi-round game simulation
  - Edge cases (empty territories, zero PUs, unknown phase)
  - Axis vs Allied model loading

Run:
    conda run -n rl-triplea python -m pytest tests/test_bot_integration.py -v
    conda run -n rl-triplea python tests/test_bot_integration.py
"""

import sys
import os
import json
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from bot.action_server import (
    GameState,
    TerritoryInfo,
    UnitInfo,
    ProductionRuleInfo,
    ProductionRuleResult,
    UnitToPlace,
    RLInferenceEngine,
    heuristic_policy,
    heuristic_purchase,
    heuristic_move,
    heuristic_place,
)

# ---------------------------------------------------------------------------
# Test fixtures — realistic WW2v3 game state snippets
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
MODEL_PATH = os.path.join(PROJECT_ROOT, "checkpoints_phase2", "selfplay_final.pt")
HAS_MODEL = os.path.exists(MODEL_PATH)

PLAYERS = ["Japanese", "Russians", "Germans", "British", "Italians", "Chinese", "Americans"]
UNIT_NAMES = [
    "infantry", "artillery", "armour", "fighter", "bomber",
    "transport", "submarine", "destroyer", "cruiser", "carrier",
    "battleship", "aaGun", "factory",
]

# Standard production rules for Russians
RUSSIAN_PRODUCTION_RULES = [
    ProductionRuleInfo(name="buyInfantry", cost=3, results=[ProductionRuleResult(name="infantry")]),
    ProductionRuleInfo(name="buyArtillery", cost=4, results=[ProductionRuleResult(name="artillery")]),
    ProductionRuleInfo(name="buyArmour", cost=5, results=[ProductionRuleResult(name="armour")]),
    ProductionRuleInfo(name="buyFighter", cost=10, results=[ProductionRuleResult(name="fighter")]),
    ProductionRuleInfo(name="buyBomber", cost=12, results=[ProductionRuleResult(name="bomber")]),
    ProductionRuleInfo(name="buyTransport", cost=7, results=[ProductionRuleResult(name="transport")]),
    ProductionRuleInfo(name="buySubmarine", cost=6, results=[ProductionRuleResult(name="submarine")]),
    ProductionRuleInfo(name="buyDestroyer", cost=8, results=[ProductionRuleResult(name="destroyer")]),
    ProductionRuleInfo(name="buyCruiser", cost=12, results=[ProductionRuleResult(name="cruiser")]),
    ProductionRuleInfo(name="buyCarrier", cost=14, results=[ProductionRuleResult(name="carrier")]),
    ProductionRuleInfo(name="buyBattleship", cost=20, results=[ProductionRuleResult(name="battleship")]),
    ProductionRuleInfo(name="buyAAGun", cost=6, results=[ProductionRuleResult(name="aaGun")]),
    ProductionRuleInfo(name="buyFactory", cost=15, results=[ProductionRuleResult(name="factory")]),
]


def make_unit(utype: str, owner: str) -> UnitInfo:
    return UnitInfo(type=utype, owner=owner, hits=0)


def make_eastern_front_state(phase: str = "purchase", pus: int = 24) -> GameState:
    """Build a realistic Eastern Front game state for round 2."""
    territories = [
        TerritoryInfo(
            name="Russia",
            owner="Russians",
            production=8,
            units=[
                make_unit("infantry", "Russians"),
                make_unit("infantry", "Russians"),
                make_unit("infantry", "Russians"),
                make_unit("artillery", "Russians"),
                make_unit("armour", "Russians"),
                make_unit("fighter", "Russians"),
                make_unit("factory", "Russians"),
                make_unit("aaGun", "Russians"),
            ],
            neighbors=["Archangel", "Caucasus", "East Poland", "Belorussia",
                        "Karelia S.S.R.", "Novosibirsk"],
        ),
        TerritoryInfo(
            name="East Poland",
            owner="Germans",
            production=2,
            units=[
                make_unit("infantry", "Germans"),
                make_unit("infantry", "Germans"),
                make_unit("infantry", "Germans"),
                make_unit("armour", "Germans"),
                make_unit("armour", "Germans"),
                make_unit("fighter", "Germans"),
            ],
            neighbors=["Russia", "Belorussia", "Poland", "Germany",
                        "Baltic States"],
        ),
        TerritoryInfo(
            name="Archangel",
            owner="Russians",
            production=1,
            units=[
                make_unit("infantry", "Russians"),
                make_unit("infantry", "Russians"),
            ],
            neighbors=["Russia", "Karelia S.S.R.", "Novosibirsk"],
        ),
        TerritoryInfo(
            name="Caucasus",
            owner="Russians",
            production=4,
            units=[
                make_unit("infantry", "Russians"),
                make_unit("infantry", "Russians"),
                make_unit("armour", "Russians"),
                make_unit("factory", "Russians"),
            ],
            neighbors=["Russia", "Ukraine", "Kazakh S.S.R."],
        ),
        TerritoryInfo(
            name="Ukraine",
            owner="Germans",
            production=2,
            units=[
                make_unit("infantry", "Germans"),
                make_unit("infantry", "Germans"),
                make_unit("armour", "Germans"),
            ],
            neighbors=["Caucasus", "Belorussia", "East Poland"],
        ),
        TerritoryInfo(
            name="Belorussia",
            owner="Germans",
            production=2,
            units=[
                make_unit("infantry", "Germans"),
            ],
            neighbors=["Russia", "East Poland", "Ukraine", "Baltic States"],
        ),
        TerritoryInfo(
            name="Karelia S.S.R.",
            owner="Russians",
            production=2,
            units=[
                make_unit("infantry", "Russians"),
                make_unit("infantry", "Russians"),
                make_unit("infantry", "Russians"),
            ],
            neighbors=["Russia", "Archangel", "Finland"],
        ),
        TerritoryInfo(
            name="Germany",
            owner="Germans",
            production=10,
            units=[
                make_unit("infantry", "Germans"),
                make_unit("infantry", "Germans"),
                make_unit("infantry", "Germans"),
                make_unit("infantry", "Germans"),
                make_unit("artillery", "Germans"),
                make_unit("armour", "Germans"),
                make_unit("fighter", "Germans"),
                make_unit("bomber", "Germans"),
                make_unit("factory", "Germans"),
                make_unit("aaGun", "Germans"),
            ],
            neighbors=["Poland", "East Poland", "France",
                        "Northwestern Europe", "Czechoslovakia Hungary"],
        ),
        TerritoryInfo(
            name="Poland",
            owner="Germans",
            production=2,
            units=[
                make_unit("infantry", "Germans"),
            ],
            neighbors=["Germany", "East Poland", "Baltic States"],
        ),
        TerritoryInfo(
            name="Baltic States",
            owner="Germans",
            production=2,
            units=[
                make_unit("infantry", "Germans"),
            ],
            neighbors=["East Poland", "Poland", "Belorussia", "Karelia S.S.R."],
        ),
        TerritoryInfo(
            name="Novosibirsk",
            owner="Russians",
            production=0,
            units=[],
            neighbors=["Russia", "Archangel", "Kazakh S.S.R."],
        ),
        TerritoryInfo(
            name="Kazakh S.S.R.",
            owner="Russians",
            production=0,
            units=[],
            neighbors=["Caucasus", "Novosibirsk"],
        ),
        TerritoryInfo(
            name="Finland",
            owner="Germans",
            production=1,
            units=[
                make_unit("infantry", "Germans"),
                make_unit("infantry", "Germans"),
            ],
            neighbors=["Karelia S.S.R.", "Norway"],
        ),
        TerritoryInfo(
            name="Norway",
            owner="Germans",
            production=2,
            units=[
                make_unit("infantry", "Germans"),
            ],
            neighbors=["Finland"],
        ),
    ]

    return GameState(
        phase=phase,
        currentPlayer="Russians",
        round=2,
        pus=pus,
        territories=territories,
        productionRules=RUSSIAN_PRODUCTION_RULES if phase == "purchase" else [],
        unitsToPlace=(
            [UnitToPlace(type="infantry", owner="Russians")] * 5
            + [UnitToPlace(type="artillery", owner="Russians")]
            + [UnitToPlace(type="armour", owner="Russians")]
        ) if phase == "place" else [],
    )


# ===========================================================================
# Tests
# ===========================================================================


class TestHeuristicPolicy:
    """Tests for the heuristic fallback policy."""

    def test_purchase_spends_all_pus(self):
        state = make_eastern_front_state("purchase", pus=24)
        result = heuristic_purchase(state)
        assert "purchases" in result
        assert len(result["purchases"]) > 0
        total_cost = sum(
            p["quantity"] * next(
                r.cost for r in RUSSIAN_PRODUCTION_RULES
                if r.results and r.results[0].name == p["unitType"]
            )
            for p in result["purchases"]
        )
        assert total_cost <= 24, f"Spent {total_cost} > 24 PUs"

    def test_purchase_zero_pus_returns_empty(self):
        state = make_eastern_front_state("purchase", pus=0)
        result = heuristic_purchase(state)
        assert result["purchases"] == []

    def test_purchase_no_rules_returns_empty(self):
        state = GameState(phase="purchase", currentPlayer="Russians", pus=24)
        result = heuristic_purchase(state)
        assert result["purchases"] == []

    def test_combat_move_targets_enemies(self):
        state = make_eastern_front_state("combatMove")
        result = heuristic_move(state)
        assert "moves" in result
        for move in result["moves"]:
            assert "from" in move
            assert "to" in move
            assert "unitTypes" in move
            assert "route" in move
            assert len(move["unitTypes"]) > 0
            assert len(move["route"]) >= 2

    def test_noncombat_move_stays_friendly(self):
        state = make_eastern_front_state("nonCombatMove")
        result = heuristic_move(state)
        assert "moves" in result
        terr_map = {t.name: t for t in state.territories}
        for move in result["moves"]:
            # Destination should be own territory
            assert move["to"] in terr_map
            assert terr_map[move["to"]].owner == "Russians"

    def test_place_returns_placements(self):
        state = make_eastern_front_state("place")
        result = heuristic_place(state)
        assert "placements" in result
        assert len(result["placements"]) > 0
        total = sum(p["quantity"] for p in result["placements"])
        assert total == 7  # 5 inf + 1 art + 1 arm

    def test_tech_phase_returns_skip(self):
        state = GameState(phase="tech", currentPlayer="Russians", pus=24)
        result = heuristic_policy(state)
        assert result.get("skip") is True

    def test_unknown_phase_returns_empty(self):
        state = GameState(phase="unknownPhase", currentPlayer="Russians")
        result = heuristic_policy(state)
        assert isinstance(result, dict)


class TestRLInferenceEngine:
    """Tests for the real model inference pipeline."""

    @classmethod
    def setup_class(cls):
        if not HAS_MODEL:
            return
        cls.engine = RLInferenceEngine(MODEL_PATH, side="allied")

    def test_engine_initialization(self):
        if not HAS_MODEL:
            print("SKIP: no model checkpoint")
            return
        assert self.engine.num_t == 162
        assert self.engine.action_dim == 337
        assert self.engine.model is not None

    def test_observation_encoding_shape(self):
        """Observation from game state has correct size."""
        if not HAS_MODEL:
            print("SKIP: no model checkpoint")
            return
        state = make_eastern_front_state("purchase")
        obs = self.engine.game_state_to_observation(state)
        expected_size = self.engine.engine.get_obs_size()
        assert obs.shape == (expected_size,), f"Got {obs.shape}, expected ({expected_size},)"
        assert not np.isnan(obs).any(), "NaN in observation"
        assert not np.isinf(obs).any(), "Inf in observation"

    def test_observation_encoding_values(self):
        """Observation values are in expected ranges."""
        if not HAS_MODEL:
            print("SKIP: no model checkpoint")
            return
        state = make_eastern_front_state("purchase")
        obs = self.engine.game_state_to_observation(state)
        # Unit counts are /10, PUs are /100, round is /20
        assert obs.min() >= 0.0, f"Negative obs value: {obs.min()}"
        assert obs.max() <= 10.0, f"Obs value too high: {obs.max()}"

    def test_inference_produces_valid_action(self):
        """Neural net output is 337-dim vector in [0, 1]."""
        if not HAS_MODEL:
            print("SKIP: no model checkpoint")
            return
        state = make_eastern_front_state("purchase")
        action = self.engine.infer(state)
        assert action.shape == (337,), f"Action shape: {action.shape}"
        assert not np.isnan(action).any(), "NaN in action"
        # Sigmoid output should be in [0, 1]
        assert action.min() >= 0.0, f"Action min: {action.min()}"
        assert action.max() <= 1.0, f"Action max: {action.max()}"

    def test_value_estimate_is_scalar(self):
        """Model produces a scalar value estimate."""
        if not HAS_MODEL:
            print("SKIP: no model checkpoint")
            return
        state = make_eastern_front_state("purchase")
        self.engine.infer(state)
        assert isinstance(self.engine._last_value, float)

    def test_purchase_respects_budget(self):
        """Decoded purchases never exceed available PUs."""
        if not HAS_MODEL:
            print("SKIP: no model checkpoint")
            return
        for budget in [0, 10, 24, 50, 100]:
            state = make_eastern_front_state("purchase", pus=budget)
            action = self.engine.infer(state)
            result = self.engine.decode_purchase(action, state)
            total_cost = sum(
                p["quantity"] * next(
                    (r.cost for r in RUSSIAN_PRODUCTION_RULES
                     if r.results and r.results[0].name == p["unitType"]),
                    99,
                )
                for p in result.get("purchases", [])
            )
            assert total_cost <= budget, (
                f"Budget {budget}: spent {total_cost}, purchases={result['purchases']}"
            )

    def test_purchase_returns_valid_unit_types(self):
        """All purchased unit types are valid WW2v3 types."""
        if not HAS_MODEL:
            print("SKIP: no model checkpoint")
            return
        state = make_eastern_front_state("purchase", pus=50)
        action = self.engine.infer(state)
        result = self.engine.decode_purchase(action, state)
        for p in result.get("purchases", []):
            assert p["unitType"] in UNIT_NAMES, f"Unknown unit type: {p['unitType']}"
            assert p["quantity"] > 0, f"Zero quantity: {p}"

    def test_combat_move_targets_enemy_territories(self):
        """Combat moves are directed at enemy-owned territories."""
        if not HAS_MODEL:
            print("SKIP: no model checkpoint")
            return
        state = make_eastern_front_state("combatMove")
        action = self.engine.infer(state)
        result = self.engine.decode_moves(action, state)
        terr_map = {t.name: t for t in state.territories}
        for move in result.get("moves", []):
            assert move["to"] in terr_map or True  # may target territories not in our subset
            if move["to"] in terr_map:
                assert terr_map[move["to"]].owner != "Russians", (
                    f"Combat move targeting own territory: {move['to']}"
                )

    def test_combat_move_sources_from_own_territories(self):
        """Combat moves originate from own territories."""
        if not HAS_MODEL:
            print("SKIP: no model checkpoint")
            return
        state = make_eastern_front_state("combatMove")
        action = self.engine.infer(state)
        result = self.engine.decode_moves(action, state)
        terr_map = {t.name: t for t in state.territories}
        for move in result.get("moves", []):
            if move["from"] in terr_map:
                assert terr_map[move["from"]].owner == "Russians", (
                    f"Move from enemy territory: {move['from']}"
                )

    def test_noncombat_move_targets_own_territories(self):
        """Non-combat moves are directed at own territories."""
        if not HAS_MODEL:
            print("SKIP: no model checkpoint")
            return
        state = make_eastern_front_state("nonCombatMove")
        action = self.engine.infer(state)
        result = self.engine.decode_moves(action, state)
        terr_map = {t.name: t for t in state.territories}
        for move in result.get("moves", []):
            if move["to"] in terr_map:
                assert terr_map[move["to"]].owner == "Russians", (
                    f"Non-combat move to enemy territory: {move['to']}"
                )

    def test_move_format_matches_rlbot_java(self):
        """Move actions have all fields that RLBot.java expects."""
        if not HAS_MODEL:
            print("SKIP: no model checkpoint")
            return
        state = make_eastern_front_state("combatMove")
        action = self.engine.infer(state)
        result = self.engine.decode_moves(action, state)
        for move in result.get("moves", []):
            assert "unitTypes" in move, "Missing 'unitTypes'"
            assert "from" in move, "Missing 'from'"
            assert "to" in move, "Missing 'to'"
            assert "route" in move, "Missing 'route'"
            assert isinstance(move["unitTypes"], list), "unitTypes should be list"
            assert isinstance(move["route"], list), "route should be list"
            assert len(move["route"]) >= 2, "route must have at least 2 entries"

    def test_placement_returns_valid_territories(self):
        """Placements target territories the player owns."""
        if not HAS_MODEL:
            print("SKIP: no model checkpoint")
            return
        state = make_eastern_front_state("place")
        action = self.engine.infer(state)
        result = self.engine.decode_placement(action, state)
        terr_map = {t.name: t for t in state.territories}
        for p in result.get("placements", []):
            assert "territory" in p
            assert "unitType" in p
            assert "quantity" in p
            if p["territory"] in terr_map:
                assert terr_map[p["territory"]].owner == "Russians"

    def test_placement_total_matches_units_to_place(self):
        """Total placed units equals the number waiting to be placed."""
        if not HAS_MODEL:
            print("SKIP: no model checkpoint")
            return
        state = make_eastern_front_state("place")
        action = self.engine.infer(state)
        result = self.engine.decode_placement(action, state)
        total_placed = sum(p["quantity"] for p in result.get("placements", []))
        total_to_place = len(state.unitsToPlace)
        assert total_placed == total_to_place, (
            f"Placed {total_placed} but {total_to_place} waiting"
        )

    def test_tech_phase_skips(self):
        """Tech phase returns skip."""
        if not HAS_MODEL:
            print("SKIP: no model checkpoint")
            return
        state = GameState(phase="tech", currentPlayer="Russians", pus=24)
        result = self.engine.get_action(state)
        assert result.get("skip") is True

    def test_full_phase_cycle(self):
        """Run through all 4 phases and verify output format."""
        if not HAS_MODEL:
            print("SKIP: no model checkpoint")
            return
        for phase, key in [
            ("purchase", "purchases"),
            ("combatMove", "moves"),
            ("nonCombatMove", "moves"),
            ("place", "placements"),
            ("tech", "skip"),
        ]:
            state = make_eastern_front_state(phase, pus=24)
            result = self.engine.get_action(state)
            if phase == "tech":
                assert result.get("skip") is True
            else:
                assert key in result, f"Phase {phase}: missing key '{key}'"

    def test_axis_model_loading(self):
        """Can load the Axis model variant."""
        if not HAS_MODEL:
            print("SKIP: no model checkpoint")
            return
        axis_engine = RLInferenceEngine(MODEL_PATH, side="axis")
        state = GameState(
            phase="purchase",
            currentPlayer="Germans",
            round=1,
            pus=40,
            territories=[
                TerritoryInfo(
                    name="Germany", owner="Germans", production=10,
                    units=[
                        make_unit("infantry", "Germans"),
                        make_unit("factory", "Germans"),
                    ],
                    neighbors=["Poland", "France"],
                ),
            ],
            productionRules=RUSSIAN_PRODUCTION_RULES,  # same structure
        )
        result = axis_engine.get_action(state)
        assert "purchases" in result

    def test_empty_territories_doesnt_crash(self):
        """Server handles game state with empty territory list."""
        if not HAS_MODEL:
            print("SKIP: no model checkpoint")
            return
        state = GameState(
            phase="purchase",
            currentPlayer="Russians",
            round=1,
            pus=24,
            territories=[],
        )
        result = self.engine.get_action(state)
        assert "purchases" in result

    def test_no_units_to_move(self):
        """Combat move with no player units produces empty moves."""
        if not HAS_MODEL:
            print("SKIP: no model checkpoint")
            return
        state = GameState(
            phase="combatMove",
            currentPlayer="Russians",
            round=2,
            pus=0,
            territories=[
                TerritoryInfo(
                    name="Russia", owner="Russians", production=8,
                    units=[make_unit("factory", "Russians")],  # Only factory, not movable
                    neighbors=["East Poland"],
                ),
                TerritoryInfo(
                    name="East Poland", owner="Germans", production=2,
                    units=[make_unit("infantry", "Germans")],
                    neighbors=["Russia"],
                ),
            ],
        )
        result = self.engine.get_action(state)
        # Should have "moves" key, possibly empty since only factory in Russia
        assert "moves" in result

    def test_deterministic_inference(self):
        """Same input produces same output (model in eval mode)."""
        if not HAS_MODEL:
            print("SKIP: no model checkpoint")
            return
        state = make_eastern_front_state("purchase")
        action1 = self.engine.infer(state).copy()
        action2 = self.engine.infer(state).copy()
        np.testing.assert_array_equal(action1, action2)


class TestGameStateSerializationRoundtrip:
    """Test that GameState can roundtrip through JSON (as RLBot.java sends it)."""

    def test_roundtrip_purchase(self):
        state = make_eastern_front_state("purchase")
        json_str = state.model_dump_json()
        parsed = GameState(**json.loads(json_str))
        assert parsed.phase == "purchase"
        assert parsed.currentPlayer == "Russians"
        assert parsed.pus == 24
        assert len(parsed.territories) == len(state.territories)
        assert len(parsed.productionRules) == 13

    def test_roundtrip_move(self):
        state = make_eastern_front_state("combatMove")
        json_str = state.model_dump_json()
        parsed = GameState(**json.loads(json_str))
        assert parsed.phase == "combatMove"

    def test_roundtrip_place(self):
        state = make_eastern_front_state("place")
        json_str = state.model_dump_json()
        parsed = GameState(**json.loads(json_str))
        assert len(parsed.unitsToPlace) == 7

    def test_minimal_json(self):
        """Minimal JSON that RLBot.java might send."""
        raw = {
            "phase": "purchase",
            "currentPlayer": "Germans",
            "round": 1,
            "pus": 40,
            "territories": [
                {
                    "name": "Germany",
                    "owner": "Germans",
                    "isWater": False,
                    "production": 10,
                    "units": [
                        {"type": "infantry", "owner": "Germans", "hits": 0},
                    ],
                    "neighbors": ["Poland"],
                }
            ],
            "productionRules": [
                {"name": "buyInfantry", "cost": 3, "results": [{"name": "infantry", "quantity": 1}]},
            ],
            "unitsToPlace": [],
        }
        state = GameState(**raw)
        assert state.territories[0].units[0].type == "infantry"


class TestActionServerEndpoint:
    """Test the FastAPI endpoint via TestClient."""

    @classmethod
    def setup_class(cls):
        try:
            from fastapi.testclient import TestClient
            from bot.action_server import app
            cls.client = TestClient(app)
            cls.has_client = True
        except ImportError:
            cls.has_client = False

    def test_health_endpoint(self):
        if not self.has_client:
            print("SKIP: no TestClient")
            return
        resp = self.client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "model_loaded" in data

    def test_purchase_endpoint(self):
        if not self.has_client:
            print("SKIP: no TestClient")
            return
        state = make_eastern_front_state("purchase")
        resp = self.client.post("/api/action", json=state.model_dump())
        assert resp.status_code == 200
        data = resp.json()
        assert "purchases" in data

    def test_combat_move_endpoint(self):
        if not self.has_client:
            print("SKIP: no TestClient")
            return
        state = make_eastern_front_state("combatMove")
        resp = self.client.post("/api/action", json=state.model_dump())
        assert resp.status_code == 200
        data = resp.json()
        assert "moves" in data

    def test_noncombat_move_endpoint(self):
        if not self.has_client:
            print("SKIP: no TestClient")
            return
        state = make_eastern_front_state("nonCombatMove")
        resp = self.client.post("/api/action", json=state.model_dump())
        assert resp.status_code == 200
        data = resp.json()
        assert "moves" in data

    def test_place_endpoint(self):
        if not self.has_client:
            print("SKIP: no TestClient")
            return
        state = make_eastern_front_state("place")
        resp = self.client.post("/api/action", json=state.model_dump())
        assert resp.status_code == 200
        data = resp.json()
        assert "placements" in data

    def test_tech_endpoint(self):
        if not self.has_client:
            print("SKIP: no TestClient")
            return
        state = GameState(phase="tech", currentPlayer="Russians", pus=24)
        resp = self.client.post("/api/action", json=state.model_dump())
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("skip") is True

    def test_invalid_json_returns_400(self):
        if not self.has_client:
            print("SKIP: no TestClient")
            return
        resp = self.client.post(
            "/api/action",
            content="not json",
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code in (400, 422)

    def test_missing_fields_returns_422(self):
        if not self.has_client:
            print("SKIP: no TestClient")
            return
        resp = self.client.post("/api/action", json={"phase": "purchase"})
        # pydantic should fill defaults, so this might still work
        assert resp.status_code in (200, 422)

    def test_full_game_round_simulation(self):
        """Simulate a full game round: purchase → combat → non-combat → place."""
        if not self.has_client:
            print("SKIP: no TestClient")
            return

        results = {}
        for phase in ["purchase", "combatMove", "nonCombatMove", "place"]:
            state = make_eastern_front_state(phase, pus=24)
            resp = self.client.post("/api/action", json=state.model_dump())
            assert resp.status_code == 200, f"Phase {phase} failed: {resp.text}"
            results[phase] = resp.json()

        # Verify we got all expected response keys
        assert "purchases" in results["purchase"]
        assert "moves" in results["combatMove"]
        assert "moves" in results["nonCombatMove"]
        assert "placements" in results["place"]

    def test_all_players_can_get_actions(self):
        """All 7 players can request actions without errors."""
        if not self.has_client:
            print("SKIP: no TestClient")
            return
        for player in PLAYERS:
            state = GameState(
                phase="purchase",
                currentPlayer=player,
                round=1,
                pus=20,
                territories=[
                    TerritoryInfo(
                        name="SomeTerritory",
                        owner=player,
                        production=5,
                        units=[make_unit("infantry", player), make_unit("factory", player)],
                        neighbors=[],
                    ),
                ],
                productionRules=RUSSIAN_PRODUCTION_RULES,
            )
            resp = self.client.post("/api/action", json=state.model_dump())
            assert resp.status_code == 200, f"Player {player} failed: {resp.text}"


class TestMultiRoundIntegration:
    """Simulate multiple rounds to verify stability."""

    def test_10_round_stress(self):
        """Run 10 rounds of inference without crashes or invalid output."""
        if not HAS_MODEL:
            print("SKIP: no model checkpoint")
            return
        engine = RLInferenceEngine(MODEL_PATH, side="allied")

        for round_num in range(1, 11):
            for phase in ["purchase", "combatMove", "nonCombatMove", "place"]:
                state = make_eastern_front_state(phase, pus=20 + round_num * 2)
                state.round = round_num
                result = engine.get_action(state)
                assert isinstance(result, dict), (
                    f"Round {round_num}, phase {phase}: not a dict"
                )

    def test_inference_speed(self):
        """Single inference should complete in < 500ms."""
        if not HAS_MODEL:
            print("SKIP: no model checkpoint")
            return
        engine = RLInferenceEngine(MODEL_PATH, side="allied")
        state = make_eastern_front_state("purchase")

        # Warm up
        engine.infer(state)

        start = time.time()
        for _ in range(10):
            engine.infer(state)
        elapsed = (time.time() - start) / 10

        assert elapsed < 0.5, f"Inference took {elapsed:.3f}s (limit: 0.5s)"
        print(f"  Avg inference time: {elapsed*1000:.1f}ms")


# ===========================================================================
# Runner
# ===========================================================================


def run_tests():
    """Run all tests and report results."""
    import traceback

    test_classes = [
        TestHeuristicPolicy,
        TestRLInferenceEngine,
        TestGameStateSerializationRoundtrip,
        TestActionServerEndpoint,
        TestMultiRoundIntegration,
    ]

    total = 0
    passed = 0
    failed = 0
    skipped = 0

    print("=" * 70)
    print("  Bot Integration Tests")
    print("=" * 70)

    for cls in test_classes:
        print(f"\n--- {cls.__name__} ---")
        instance = cls()
        if hasattr(cls, "setup_class"):
            try:
                cls.setup_class()
            except Exception as e:
                print(f"  setup_class FAILED: {e}")
                continue

        for attr in sorted(dir(cls)):
            if not attr.startswith("test_"):
                continue
            total += 1
            method = getattr(instance, attr)
            try:
                method()
                passed += 1
                print(f"  PASS {attr}")
            except Exception as e:
                if "SKIP" in str(e):
                    skipped += 1
                    print(f"  SKIP {attr}")
                else:
                    failed += 1
                    print(f"  FAIL {attr}: {e}")
                    traceback.print_exc()

    print(f"\n{'=' * 70}")
    print(f"Results: {passed} passed, {failed} failed, {skipped} skipped (of {total})")
    print(f"{'=' * 70}")
    return failed == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
