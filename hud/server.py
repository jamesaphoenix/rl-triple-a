#!/usr/bin/env python3
"""TripleA HUD Recommender Server.

Watches TripleA autosave directory for changes, extracts game state,
runs the trained neural net, and serves recommendations via web UI.

Usage:
    python hud/server.py --model checkpoints_selfplay_v3/selfplay_final.pt

Then open http://localhost:8080 in your browser while playing TripleA.
"""

import sys
import os
import json
import time
import gzip
import struct
import hashlib
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler
from threading import Thread
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch

from src.game_data_export import export_map_arrays
from src.network_v2 import ActorCriticV2
from triplea_engine import TripleAEngine

# ── Game State Extraction from .tsvg ────────────────────────

PLAYERS = ["Japanese", "Russians", "Germans", "British", "Italians", "Chinese", "Americans"]
PLAYER_IDX = {p: i for i, p in enumerate(PLAYERS)}
UNIT_NAMES = ["infantry", "artillery", "armour", "fighter", "bomber",
              "transport", "submarine", "destroyer", "cruiser", "carrier",
              "battleship", "aaGun", "factory"]
AXIS_PLAYERS = {"Japanese", "Germans", "Italians"}
ALLIED_PLAYERS = {"Russians", "British", "Chinese", "Americans"}

NUM_UNIT_TYPES = 13
NUM_PLAYERS = 7


def detect_phase_from_filename(filename: str) -> dict:
    """Parse autosave filename to determine game phase."""
    name = Path(filename).stem
    phase_map = {
        "autosaveAfterJapaneseCombatMove": ("Japanese", "after_combat_move"),
        "autosaveAfterJapaneseNonCombatMove": ("Japanese", "after_noncombat_move"),
        "autosaveAfterRussianCombatMove": ("Russians", "after_combat_move"),
        "autosaveAfterRussianNonCombatMove": ("Russians", "after_noncombat_move"),
        "autosaveAfterGermanCombatMove": ("Germans", "after_combat_move"),
        "autosaveAfterGermanNonCombatMove": ("Germans", "after_noncombat_move"),
        "autosaveAfterBritishCombatMove": ("British", "after_combat_move"),
        "autosaveAfterBritishNonCombatMove": ("British", "after_noncombat_move"),
        "autosaveAfterItalianCombatMove": ("Italians", "after_combat_move"),
        "autosaveAfterItalianNonCombatMove": ("Italians", "after_noncombat_move"),
        "autosaveAfterAmericanCombatMove": ("Americans", "after_combat_move"),
        "autosaveAfterAmericanNonCombatMove": ("Americans", "after_noncombat_move"),
        "autosaveAfterChineseCombatMove": ("Chinese", "after_combat_move"),
        "autosaveAfterChineseNonCombatMove": ("Chinese", "after_noncombat_move"),
        "autosaveAfterBattle": (None, "after_battle"),
        "autosaveBeforeBattle": (None, "before_battle"),
        "autosaveBeforeEndTurn": (None, "before_end_turn"),
        "autosave_round_even": (None, "round_even"),
        "autosave_round_odd": (None, "round_odd"),
    }
    if name in phase_map:
        return {"player": phase_map[name][0], "phase": phase_map[name][1]}
    return {"player": None, "phase": "unknown"}


def get_next_allied_player(phase_info: dict) -> str:
    """Determine which Allied player needs recommendations next."""
    turn_order = ["Japanese", "Russians", "Germans", "British", "Italians", "Chinese", "Americans"]
    player = phase_info.get("player")
    phase = phase_info.get("phase", "")

    if player and player in AXIS_PLAYERS and "noncombat" in phase:
        # Axis just finished — next Allied player is coming
        idx = turn_order.index(player)
        for i in range(1, len(turn_order)):
            next_p = turn_order[(idx + i) % len(turn_order)]
            if next_p in ALLIED_PLAYERS:
                return next_p
        return "Russians"

    if player and player in ALLIED_PLAYERS:
        return player

    # Default
    return "Russians"


# ── Neural Net Recommender ────────────────────────────────────

class Recommender:
    def __init__(self, model_path: str):
        self.arrays = export_map_arrays()
        self.territory_names = self.arrays["territory_names"]
        self.tidx = {n: i for i, n in enumerate(self.territory_names)}
        self.num_t = len(self.territory_names)

        # Create engine for observation encoding
        self.engine = TripleAEngine(
            self.arrays["adjacency"], self.arrays["is_water"],
            self.arrays["is_impassable"], self.arrays["production"],
            self.arrays["is_victory_city"], self.arrays["is_capital"],
            self.arrays["chinese_territories"],
            self.arrays["initial_units"], self.arrays["initial_owner"],
            self.arrays["initial_pus"], seed=0,
        )
        for no in self.arrays["national_objectives"]:
            self.engine.add_national_objective(
                no["player"], no["value"], no["territories"],
                no["count"], no["enemy_sea_zones"], no.get("allied_exclusion", False),
            )

        obs_size = self.engine.get_obs_size()
        action_dim = NUM_UNIT_TYPES + self.num_t + self.num_t

        # Load model
        self.device = torch.device("cpu")  # CPU for inference
        self.model = ActorCriticV2(obs_size, action_dim, hidden_size=512).to(self.device)

        if model_path and Path(model_path).exists():
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
            if "allied" in checkpoint:
                self.model.load_state_dict(checkpoint["allied"])
                print(f"Loaded Allied model from {model_path}")
            elif "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
                print(f"Loaded model from {model_path}")
            else:
                self.model.load_state_dict(checkpoint)
                print(f"Loaded model from {model_path}")
        else:
            print(f"WARNING: No model loaded — using random weights")

        self.model.eval()

    def get_recommendations(self, player: str, game_state: dict = None) -> dict:
        """Generate recommendations for an Allied player.

        Args:
            player: which Allied player ('Russians', 'British', etc.)
            game_state: parsed JSON from Java extractor (live game state)
        """
        budget = 24  # default
        game_round = 1

        if game_state:
            # Load live game state into engine
            tidx = self.tidx
            T = self.num_t
            owners = np.full(T, -1, dtype=np.int32)
            units = np.zeros((T, NUM_PLAYERS, NUM_UNIT_TYPES), dtype=np.int32)
            pus = np.zeros(NUM_PLAYERS, dtype=np.int32)

            for name, t in game_state.get("territories", {}).items():
                if name not in tidx: continue
                i = tidx[name]
                if t.get("owner") and t["owner"] in PLAYER_IDX:
                    owners[i] = PLAYER_IDX[t["owner"]]
                for oname, umap in t.get("units", {}).items():
                    if oname not in PLAYER_IDX: continue
                    pi = PLAYER_IDX[oname]
                    for uname, count in umap.items():
                        if uname in {"infantry": 0, "artillery": 1, "armour": 2,
                                     "fighter": 3, "bomber": 4, "transport": 5,
                                     "submarine": 6, "destroyer": 7, "cruiser": 8,
                                     "carrier": 9, "battleship": 10, "aaGun": 11,
                                     "factory": 12}:
                            pass
                        ui = {"infantry": 0, "artillery": 1, "armour": 2,
                              "fighter": 3, "bomber": 4, "transport": 5,
                              "submarine": 6, "destroyer": 7, "cruiser": 8,
                              "carrier": 9, "battleship": 10, "aaGun": 11,
                              "factory": 12}.get(uname, -1)
                        if ui >= 0:
                            units[i, pi, ui] = count

            for pname, pd in game_state.get("players", {}).items():
                if pname in PLAYER_IDX:
                    pus[PLAYER_IDX[pname]] = pd.get("pus", 0)

            p_idx = PLAYER_IDX.get(player, 1)
            budget = int(pus[p_idx])
            game_round = game_state.get("round", 1)

            obs = np.array(self.engine.load_state(owners, units, pus, game_round, p_idx), dtype=np.float32)
        else:
            obs = np.array(self.engine.reset(42), dtype=np.float32)
        obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action_mean, value = self.model.forward(obs_tensor)
            action = action_mean.squeeze(0).cpu().numpy()

        purchase_scores = action[:NUM_UNIT_TYPES]
        attack_scores = action[NUM_UNIT_TYPES:NUM_UNIT_TYPES + self.num_t]
        reinforce_scores = action[NUM_UNIT_TYPES + self.num_t:]

        # Build live unit map if we have game state
        # units_by_terr[territory_name] = {unit_name: count}
        units_by_terr = {}
        owners_map = {}
        if game_state:
            for tname, tdata in game_state.get("territories", {}).items():
                owners_map[tname] = tdata.get("owner")
                if player in tdata.get("units", {}):
                    units_by_terr[tname] = tdata["units"][player]

        p_idx = PLAYER_IDX.get(player, 1)
        is_axis_player = p_idx in (0, 2, 4)

        # ── PURCHASE: budget-constrained, exact unit counts ──
        unit_costs = {"infantry": 3, "artillery": 4, "armour": 5, "fighter": 10,
                      "bomber": 12, "transport": 7, "submarine": 6, "destroyer": 8,
                      "cruiser": 12, "carrier": 14, "battleship": 20, "aaGun": 6, "factory": 15}
        purchase_recs = []
        remaining_budget = budget
        sorted_units = sorted(enumerate(purchase_scores), key=lambda x: -x[1])
        for idx, score in sorted_units:
            if score < 0.05 or remaining_budget <= 0:
                break
            uname = UNIT_NAMES[idx]
            cost = unit_costs.get(uname, 99)
            count = min(int(score * 20), remaining_budget // cost)
            if count > 0:
                purchase_recs.append({
                    "unit": uname,
                    "count": count,
                    "cost_each": cost,
                    "total_cost": count * cost,
                    "action": f"Buy {count}x {uname} ({count * cost} PUs)",
                })
                remaining_budget -= count * cost

        # ── COMBAT MOVE: exact unit orders from adjacent territories ──
        attack_recs = []
        adj = self.arrays["adjacency"]
        for t_idx in np.argsort(-attack_scores):
            score = attack_scores[t_idx]
            if score < 0.3:
                break
            tname = self.territory_names[t_idx]
            if self.arrays["is_impassable"][t_idx]:
                continue
            # Check this territory is enemy-owned
            t_owner = owners_map.get(tname)
            if t_owner == player:
                continue  # don't attack own territory

            # Find our units in adjacent territories
            move_orders = []
            for n_idx in range(self.num_t):
                if not adj[t_idx, n_idx]:
                    continue
                n_name = self.territory_names[n_idx]
                if n_name not in units_by_terr:
                    continue
                units_here = units_by_terr[n_name]
                # List movable combat units
                movable = []
                for uname, count in units_here.items():
                    if uname in ("factory", "aaGun"):
                        continue
                    # Keep 1 infantry for defense if territory has production
                    keep = 1 if uname == "infantry" and self.arrays["production"][self.tidx[n_name]] >= 2 else 0
                    send = max(0, count - keep)
                    if send > 0:
                        movable.append(f"{send}x {uname}")
                if movable:
                    move_orders.append({
                        "from": n_name,
                        "units": ", ".join(movable),
                        "action": f"Move {', '.join(movable)} from {n_name} → {tname}",
                    })

            if move_orders:
                attack_recs.append({
                    "territory": tname,
                    "owner": t_owner,
                    "production": int(self.arrays["production"][t_idx]),
                    "priority": float(score),
                    "orders": move_orders,
                    "action": f"Attack {tname}" + (f" ({t_owner})" if t_owner else ""),
                })
            if len(attack_recs) >= 8:
                break

        # ── NON-COMBAT: specific reinforcement moves ──
        reinforce_recs = []
        for t_idx in np.argsort(-reinforce_scores):
            score = reinforce_scores[t_idx]
            if score < 0.3:
                break
            tname = self.territory_names[t_idx]
            if self.arrays["is_impassable"][t_idx] or self.arrays["is_water"][t_idx]:
                continue
            t_owner = owners_map.get(tname)
            if t_owner != player:
                continue  # only reinforce own territories

            move_orders = []
            for n_idx in range(self.num_t):
                if not adj[t_idx, n_idx]:
                    continue
                n_name = self.territory_names[n_idx]
                if n_name not in units_by_terr:
                    continue
                n_score = reinforce_scores[self.tidx.get(n_name, 0)] if n_name in self.tidx else 0
                if n_score >= score:
                    continue  # don't pull from higher-priority territory
                units_here = units_by_terr[n_name]
                movable = []
                for uname, count in units_here.items():
                    if uname in ("factory", "aaGun", "fighter", "bomber"):
                        continue
                    keep = 1 if uname == "infantry" and self.arrays["production"][self.tidx[n_name]] > 0 else 0
                    send = max(0, count - keep)
                    if send > 0:
                        movable.append(f"{send}x {uname}")
                if movable:
                    move_orders.append({
                        "from": n_name,
                        "units": ", ".join(movable),
                        "action": f"Move {', '.join(movable)} from {n_name} → {tname}",
                    })

            if move_orders:
                reinforce_recs.append({
                    "territory": tname,
                    "priority": float(score),
                    "orders": move_orders,
                    "action": f"Reinforce {tname}",
                })
            if len(reinforce_recs) >= 8:
                break

        # ── PLACEMENT: where to put purchased units ──
        placement_recs = []
        if game_state and purchase_recs:
            # Find player's factories
            factories = []
            for tname, tdata in game_state.get("territories", {}).items():
                if tdata.get("owner") == player:
                    player_units = tdata.get("units", {}).get(player, {})
                    if player_units.get("factory", 0) > 0:
                        prod = self.arrays["production"][self.tidx[tname]] if tname in self.tidx else 0
                        factories.append({"territory": tname, "production": prod})
            factories.sort(key=lambda x: -x["production"])

            bought_str = ", ".join(r["action"].replace("Buy ", "") for r in purchase_recs)
            if factories:
                main_factory = factories[0]["territory"]
                placement_recs.append({
                    "territory": main_factory,
                    "action": f"Place {bought_str} at {main_factory}",
                    "detail": f"Factory capacity: {factories[0]['production']} units",
                })
                for f in factories[1:]:
                    placement_recs.append({
                        "territory": f["territory"],
                        "action": f"Overflow placement at {f['territory']} if {main_factory} is full",
                        "detail": f"Factory capacity: {f['production']} units",
                    })

        return {
            "player": player,
            "budget": budget,
            "budget_remaining": remaining_budget,
            "round": game_round,
            "estimated_value": float(value.item()),
            "purchase": purchase_recs[:8],
            "attacks": attack_recs[:10],
            "reinforce": reinforce_recs[:10],
            "placement": placement_recs,
            "timestamp": time.time(),
        }


# ── File Watcher ──────────────────────────────────────────────

class SaveFileHandler(FileSystemEventHandler):
    def __init__(self, recommender, state):
        self.recommender = recommender
        self.state = state
        self.last_hash = ""

    def on_modified(self, event):
        if not event.src_path.endswith(".tsvg"):
            return
        # Debounce: check if file actually changed
        try:
            with open(event.src_path, "rb") as f:
                h = hashlib.md5(f.read(1024)).hexdigest()
            if h == self.last_hash:
                return
            self.last_hash = h
        except:
            return

        filename = Path(event.src_path).name
        phase_info = detect_phase_from_filename(filename)
        next_player = get_next_allied_player(phase_info)

        print(f"[{time.strftime('%H:%M:%S')}] Save detected: {filename}")
        print(f"  Phase: {phase_info['phase']} | Next Allied: {next_player}")

        # Extract live game state via Java extractor
        game_state = None
        try:
            import subprocess, tempfile, json as json_mod
            with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
                tmp_path = tmp.name
            extract_script = str(Path(__file__).parent.parent / "tools" / "extract_live.sh")
            result = subprocess.run([extract_script, event.src_path, tmp_path],
                                   capture_output=True, timeout=15)
            if result.returncode == 0:
                with open(tmp_path) as f:
                    game_state = json_mod.load(f)
                print(f"  Live state extracted: Round {game_state.get('round', '?')}")
            os.unlink(tmp_path)
        except Exception as e:
            print(f"  Warning: could not extract live state: {e}")

        recs = self.recommender.get_recommendations(next_player, game_state)
        self.state["recommendations"] = recs
        self.state["last_save"] = filename
        self.state["last_update"] = time.strftime("%H:%M:%S")

        # Print summary
        buy_str = ", ".join(str(r["count"]) + "x " + r["unit"] for r in recs["purchase"][:5])
        print(f"  Purchase: {buy_str}")
        if recs["attacks"]:
            atk_str = ", ".join(r["territory"] for r in recs["attacks"][:5])
            print(f"  Attack: {atk_str}")
        if recs["reinforce"]:
            print(f"  Reinforce: {', '.join(r['territory'] for r in recs['reinforce'][:5])}")


# ── Web Server ────────────────────────────────────────────────

SHARED_STATE = {"recommendations": None, "last_save": None, "last_update": None}


class HUDHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/api/recommendations":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(SHARED_STATE).encode())
        elif self.path == "/" or self.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            html_path = Path(__file__).parent / "index.html"
            self.wfile.write(html_path.read_bytes())
        else:
            super().do_GET()

    def log_message(self, format, *args):
        pass  # Silence request logging


# ── Main ──────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="TripleA HUD Recommender")
    parser.add_argument("--model", type=str, default="checkpoints_selfplay_v3/selfplay_final.pt")
    parser.add_argument("--watch-dir", type=str,
                       default=str(Path.home() / "triplea" / "savedGames" / "autoSave"))
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    print("=" * 60)
    print("  TripleA HUD Recommender")
    print(f"  Model: {args.model}")
    print(f"  Watching: {args.watch_dir}")
    print(f"  Web UI: http://localhost:{args.port}")
    print("=" * 60)

    recommender = Recommender(args.model)

    # Start file watcher
    handler = SaveFileHandler(recommender, SHARED_STATE)
    observer = Observer()
    observer.schedule(handler, args.watch_dir, recursive=False)
    observer.start()
    print(f"Watching {args.watch_dir} for save file changes...")

    # Generate initial recommendations
    recs = recommender.get_recommendations("Russians")
    SHARED_STATE["recommendations"] = recs
    SHARED_STATE["last_update"] = time.strftime("%H:%M:%S")

    # Start web server
    server = HTTPServer(("localhost", args.port), HUDHandler)
    print(f"HUD running at http://localhost:{args.port}")
    print("Open this in your browser while playing TripleA\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


if __name__ == "__main__":
    main()
