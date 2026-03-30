#!/usr/bin/env python3
"""Integration tests for the live game state extraction pipeline.

Tests the full flow:
1. Java extractor reads .tsvg save file → JSON
2. Python parses JSON → numpy arrays
3. Rust engine loads state via load_state()
4. Neural net generates recommendations from live state
"""

import sys
import os
import json
import subprocess
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

SAVE_DIR = os.path.expanduser("~/triplea/savedGames/autoSave")
EXTRACT_SCRIPT = os.path.join(os.path.dirname(__file__), "..", "tools", "extract_live.sh")
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")

PLAYERS = ["Japanese", "Russians", "Germans", "British", "Italians", "Chinese", "Americans"]
PLAYER_IDX = {p: i for i, p in enumerate(PLAYERS)}
UNIT_NAMES = ["infantry", "artillery", "armour", "fighter", "bomber",
              "transport", "submarine", "destroyer", "cruiser", "carrier",
              "battleship", "aaGun", "factory"]
UNIT_IDX = {n: i for i, n in enumerate(UNIT_NAMES)}
NUM_PLAYERS = 7
NUM_UNIT_TYPES = 13


def test_java_extractor():
    """Test 1: Java extractor produces valid JSON from a .tsvg file."""
    save_file = os.path.join(SAVE_DIR, "autosave_round_even.tsvg")
    if not os.path.exists(save_file):
        print("SKIP test_java_extractor: no save file found")
        return True

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        out_path = f.name

    result = subprocess.run(
        [EXTRACT_SCRIPT, save_file, out_path],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, f"Extractor failed: {result.stderr}"

    with open(out_path) as f:
        data = json.load(f)

    os.unlink(out_path)

    # Validate structure
    assert "gameName" in data, "Missing gameName"
    assert "round" in data, "Missing round"
    assert "players" in data, "Missing players"
    assert "territories" in data, "Missing territories"
    assert data["round"] > 0, f"Invalid round: {data['round']}"
    assert len(data["players"]) == 7, f"Expected 7 players, got {len(data['players'])}"
    assert len(data["territories"]) == 162, f"Expected 162 territories, got {len(data['territories'])}"

    # Validate player PUs
    for player in PLAYERS:
        assert player in data["players"], f"Missing player: {player}"
        assert "pus" in data["players"][player], f"Missing PUs for {player}"
        assert data["players"][player]["pus"] >= 0, f"Negative PUs for {player}"

    # Validate territory structure
    for name, t in data["territories"].items():
        assert "owner" in t, f"Missing owner for {name}"
        assert "isWater" in t, f"Missing isWater for {name}"
        assert "units" in t, f"Missing units for {name}"

    print(f"PASS test_java_extractor: Round {data['round']}, "
          f"{sum(1 for t in data['territories'].values() if t['units'])} territories with units")
    return True


def test_json_to_numpy():
    """Test 2: JSON game state converts to numpy arrays matching Rust engine format."""
    save_file = os.path.join(SAVE_DIR, "autosave_round_even.tsvg")
    if not os.path.exists(save_file):
        print("SKIP test_json_to_numpy: no save file found")
        return True

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        out_path = f.name

    subprocess.run([EXTRACT_SCRIPT, save_file, out_path],
                   capture_output=True, timeout=30)

    with open(out_path) as f:
        data = json.load(f)
    os.unlink(out_path)

    # Convert to numpy arrays
    from src.game_data_export import export_map_arrays
    arrays = export_map_arrays()
    territory_names = arrays["territory_names"]
    tidx = {n: i for i, n in enumerate(territory_names)}
    T = len(territory_names)

    owners = np.full(T, -1, dtype=np.int32)
    units = np.zeros((T, NUM_PLAYERS, NUM_UNIT_TYPES), dtype=np.int32)
    pus = np.zeros(NUM_PLAYERS, dtype=np.int32)

    for name, t in data["territories"].items():
        if name not in tidx:
            continue
        i = tidx[name]
        if t["owner"] and t["owner"] in PLAYER_IDX:
            owners[i] = PLAYER_IDX[t["owner"]]
        for owner_name, unit_map in t.get("units", {}).items():
            if owner_name not in PLAYER_IDX:
                continue
            pi = PLAYER_IDX[owner_name]
            for unit_name, count in unit_map.items():
                if unit_name in UNIT_IDX:
                    units[i, pi, UNIT_IDX[unit_name]] = count

    for player_name, pdata in data["players"].items():
        if player_name in PLAYER_IDX:
            pus[PLAYER_IDX[player_name]] = pdata["pus"]

    # Validate shapes
    assert owners.shape == (T,), f"owners shape: {owners.shape}"
    assert units.shape == (T, NUM_PLAYERS, NUM_UNIT_TYPES), f"units shape: {units.shape}"
    assert pus.shape == (NUM_PLAYERS,), f"pus shape: {pus.shape}"

    # Validate content
    total_units = units.sum()
    assert total_units > 0, "No units found"
    assert (owners >= -1).all(), "Invalid owner values"
    assert (pus >= 0).all(), "Negative PUs"

    print(f"PASS test_json_to_numpy: {total_units} total units, "
          f"owners set for {(owners >= 0).sum()} territories")
    return True


def test_load_state_into_engine():
    """Test 3: Rust engine accepts live game state and produces valid observation."""
    save_file = os.path.join(SAVE_DIR, "autosave_round_even.tsvg")
    if not os.path.exists(save_file):
        print("SKIP test_load_state_into_engine: no save file found")
        return True

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        out_path = f.name

    subprocess.run([EXTRACT_SCRIPT, save_file, out_path],
                   capture_output=True, timeout=30)

    with open(out_path) as f:
        data = json.load(f)
    os.unlink(out_path)

    from src.game_data_export import export_map_arrays
    from triplea_engine import TripleAEngine

    arrays = export_map_arrays()
    territory_names = arrays["territory_names"]
    tidx = {n: i for i, n in enumerate(territory_names)}
    T = len(territory_names)

    # Build state arrays from JSON
    owners = np.full(T, -1, dtype=np.int32)
    units = np.zeros((T, NUM_PLAYERS, NUM_UNIT_TYPES), dtype=np.int32)
    pus = np.zeros(NUM_PLAYERS, dtype=np.int32)

    for name, t in data["territories"].items():
        if name not in tidx: continue
        i = tidx[name]
        if t["owner"] and t["owner"] in PLAYER_IDX:
            owners[i] = PLAYER_IDX[t["owner"]]
        for owner_name, unit_map in t.get("units", {}).items():
            if owner_name not in PLAYER_IDX: continue
            pi = PLAYER_IDX[owner_name]
            for unit_name, count in unit_map.items():
                if unit_name in UNIT_IDX:
                    units[i, pi, UNIT_IDX[unit_name]] = count

    for player_name, pdata in data["players"].items():
        if player_name in PLAYER_IDX:
            pus[PLAYER_IDX[player_name]] = pdata["pus"]

    # Create engine and load state
    engine = TripleAEngine(
        arrays["adjacency"], arrays["is_water"], arrays["is_impassable"],
        arrays["production"], arrays["is_victory_city"], arrays["is_capital"],
        arrays["chinese_territories"],
        arrays["initial_units"], arrays["initial_owner"], arrays["initial_pus"],
        seed=42,
    )

    obs = np.array(engine.load_state(owners, units, pus, data["round"], 1))

    assert len(obs) == engine.get_obs_size(), f"Obs size mismatch: {len(obs)} vs {engine.get_obs_size()}"
    assert not np.isnan(obs).any(), "NaN in observation"
    assert not np.isinf(obs).any(), "Inf in observation"
    assert obs.max() <= 10.0, f"Obs max too high: {obs.max()}"

    print(f"PASS test_load_state_into_engine: obs size {len(obs)}, "
          f"range [{obs.min():.3f}, {obs.max():.3f}]")
    return True


def test_neural_net_on_live_state():
    """Test 4: Neural net produces valid recommendations from live game state."""
    save_file = os.path.join(SAVE_DIR, "autosave_round_even.tsvg")
    model_path = None
    for p in ["checkpoints_selfplay_v3/foundation_v1_282k_games_93pct.pt",
              "checkpoints_selfplay_v3/selfplay_final.pt",
              "checkpoints_phase2/selfplay_100.pt"]:
        full = os.path.join(PROJECT_ROOT, p)
        if os.path.exists(full):
            model_path = full
            break

    if not os.path.exists(save_file) or not model_path:
        print("SKIP test_neural_net_on_live_state: missing save file or model")
        return True

    import torch
    from src.network_v2 import ActorCriticV2
    from src.game_data_export import export_map_arrays
    from triplea_engine import TripleAEngine

    # Extract game state
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        out_path = f.name
    subprocess.run([EXTRACT_SCRIPT, save_file, out_path],
                   capture_output=True, timeout=30)
    with open(out_path) as f:
        data = json.load(f)
    os.unlink(out_path)

    # Build arrays
    arrays = export_map_arrays()
    territory_names = arrays["territory_names"]
    tidx = {n: i for i, n in enumerate(territory_names)}
    T = len(territory_names)

    owners = np.full(T, -1, dtype=np.int32)
    units_arr = np.zeros((T, NUM_PLAYERS, NUM_UNIT_TYPES), dtype=np.int32)
    pus_arr = np.zeros(NUM_PLAYERS, dtype=np.int32)

    for name, t in data["territories"].items():
        if name not in tidx: continue
        i = tidx[name]
        if t["owner"] and t["owner"] in PLAYER_IDX:
            owners[i] = PLAYER_IDX[t["owner"]]
        for owner_name, unit_map in t.get("units", {}).items():
            if owner_name not in PLAYER_IDX: continue
            pi = PLAYER_IDX[owner_name]
            for unit_name, count in unit_map.items():
                if unit_name in UNIT_IDX:
                    units_arr[i, pi, UNIT_IDX[unit_name]] = count

    for player_name, pdata in data["players"].items():
        if player_name in PLAYER_IDX:
            pus_arr[PLAYER_IDX[player_name]] = pdata["pus"]

    # Load into engine
    engine = TripleAEngine(
        arrays["adjacency"], arrays["is_water"], arrays["is_impassable"],
        arrays["production"], arrays["is_victory_city"], arrays["is_capital"],
        arrays["chinese_territories"],
        arrays["initial_units"], arrays["initial_owner"], arrays["initial_pus"],
        seed=42,
    )
    obs = np.array(engine.load_state(owners, units_arr, pus_arr, data["round"], 1))

    # Load model
    obs_size = engine.get_obs_size()
    action_dim = NUM_UNIT_TYPES + T + T
    model = ActorCriticV2(obs_size, action_dim, hidden_size=512)
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    if "allied" in ckpt:
        model.load_state_dict(ckpt["allied"])
    model.eval()

    # Run inference
    obs_tensor = torch.from_numpy(obs).unsqueeze(0)
    with torch.no_grad():
        action_mean, value = model.forward(obs_tensor)
    action = action_mean.squeeze(0).numpy()

    assert action.shape == (action_dim,), f"Action shape: {action.shape}"
    assert not np.isnan(action).any(), "NaN in action"
    assert (action >= 0).all() and (action <= 1).all(), f"Action out of [0,1]: [{action.min()}, {action.max()}]"

    # Extract recommendations
    purchases = action[:NUM_UNIT_TYPES]
    attacks = action[NUM_UNIT_TYPES:NUM_UNIT_TYPES + T]
    reinforces = action[NUM_UNIT_TYPES + T:]

    top_purchase = sorted(enumerate(purchases), key=lambda x: -x[1])[:3]
    top_attack = sorted(enumerate(attacks), key=lambda x: -x[1])[:3]
    top_reinforce = sorted(enumerate(reinforces), key=lambda x: -x[1])[:3]

    print(f"PASS test_neural_net_on_live_state:")
    print(f"  Game round: {data['round']}")
    print(f"  Position value: {value.item():.2f}")
    print(f"  Top purchases: {[(UNIT_NAMES[i], f'{s:.2f}') for i, s in top_purchase]}")
    print(f"  Top attacks: {[(territory_names[i], f'{s:.2f}') for i, s in top_attack if not arrays['is_water'][i]]}")
    print(f"  Top reinforcements: {[(territory_names[i], f'{s:.2f}') for i, s in top_reinforce if not arrays['is_water'][i]]}")
    return True


def test_all_save_files_extractable():
    """Test 5: All autosave files can be extracted without errors."""
    import glob
    save_files = glob.glob(os.path.join(SAVE_DIR, "*.tsvg"))
    if not save_files:
        print("SKIP test_all_save_files_extractable: no save files")
        return True

    passed = 0
    failed = 0
    for sf in save_files:
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            out_path = f.name
        result = subprocess.run([EXTRACT_SCRIPT, sf, out_path],
                               capture_output=True, timeout=30)
        if result.returncode == 0:
            try:
                with open(out_path) as f:
                    data = json.load(f)
                assert "territories" in data
                passed += 1
            except:
                failed += 1
                print(f"  FAIL: {os.path.basename(sf)} — invalid JSON")
        else:
            failed += 1
            print(f"  FAIL: {os.path.basename(sf)} — extraction error")
        os.unlink(out_path)

    print(f"PASS test_all_save_files_extractable: {passed}/{passed+failed} files OK")
    return failed == 0


if __name__ == "__main__":
    print("=" * 60)
    print("  Integration Tests: Live Game State Pipeline")
    print("=" * 60)

    results = []
    for test_fn in [test_java_extractor, test_json_to_numpy,
                    test_load_state_into_engine, test_neural_net_on_live_state,
                    test_all_save_files_extractable]:
        try:
            results.append(test_fn())
        except Exception as e:
            print(f"FAIL {test_fn.__name__}: {e}")
            results.append(False)

    print()
    passed = sum(1 for r in results if r)
    print(f"Results: {passed}/{len(results)} passed")
    sys.exit(0 if all(results) else 1)
