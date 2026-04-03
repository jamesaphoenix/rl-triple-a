#!/usr/bin/env python3
"""
RL Bot Log Collector — extracts, parses, and deduplicates game session errors.

Reads from:
  - ~/triplea/triplea.log  (TripleA client logs with RlBotAi messages)
  - Action server stdout/stderr (if redirected to a log file)

Outputs:
  - experiments/game_logs/<timestamp>_session.json  (structured per-session)
  - experiments/game_logs/error_registry.json       (deduped error catalog)

Usage:
    python bot/log_collector.py                     # one-shot parse
    python bot/log_collector.py --watch             # tail and parse continuously
    python bot/log_collector.py --summary           # print deduped error summary
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRIPLEA_LOG = Path.home() / "triplea" / "triplea.log"
GAME_LOGS_DIR = PROJECT_ROOT / "experiments" / "game_logs"
ERROR_REGISTRY = GAME_LOGS_DIR / "error_registry.json"

# ---------------------------------------------------------------------------
# Error classification patterns
# ---------------------------------------------------------------------------

ERROR_PATTERNS = [
    {
        "category": "impassable_territory",
        "pattern": re.compile(r"RL move (.+?) -> (.+?) failed: Can't move through impassable territories"),
        "engine_fix": "Ensure action masking blocks moves targeting impassable territories",
        "rust_area": "execute_combat / execute_reinforce — adjacency filtering",
    },
    {
        "category": "no_transports",
        "pattern": re.compile(r"RL move (.+?) -> (.+?) failed: Not enough transports"),
        "engine_fix": "Land-to-sea moves must validate transport availability in the action space",
        "rust_area": "execute_combat / execute_reinforce — amphibious logic",
    },
    {
        "category": "hostile_sea_loading",
        "pattern": re.compile(r"RL move (.+?) -> (.+?) failed: Cannot load when enemy sea units are present"),
        "engine_fix": "Block transport loading in sea zones containing enemy warships",
        "rust_area": "execute_combat — amphibious source validation",
    },
    {
        "category": "chinese_restrictions",
        "pattern": re.compile(r"RL move (.+?) -> (.+?) failed: Cannot move outside restricted territories"),
        "engine_fix": "Chinese units must be masked to only target chinese_territories",
        "rust_area": "is_valid_chinese_move + action masking in execute_combat/reinforce",
    },
    {
        "category": "placement_invalid",
        "pattern": re.compile(r"RL placement at (.+?) failed: (.+)"),
        "engine_fix": "Placement must validate: territory owned, has factory, not conquered this turn, capacity",
        "rust_area": "execute_placement — factory ownership and capacity checks",
    },
    {
        "category": "move_generic",
        "pattern": re.compile(r"RL move (.+?) -> (.+?) failed: (.+)"),
        "engine_fix": "Generic move failure — investigate specific rule",
        "rust_area": "movement validation",
    },
]

# Info-level patterns (not errors but useful signals)
INFO_PATTERNS = [
    {
        "category": "phase_received",
        "pattern": re.compile(r"Received (\w+) phase for '(.+?)' \(round (\d+), (\d+) PUs, (\d+) territories\)"),
    },
    {
        "category": "server_response",
        "pattern": re.compile(r"Returning: (.+)"),
    },
]


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_rl_log_line(line: str) -> dict | None:
    """Parse a single log line into a structured record, or None if not RL-related."""
    # Check for RlBotAi lines
    if "org.triplea.ai.rl.RlBotAi" not in line and "rl_action_server" not in line:
        return None

    # Extract timestamp
    ts_match = re.match(r"(\d+)\s+(\d{2}:\d{2}:\d{2}\.\d{3})", line)
    timestamp = ts_match.group(2) if ts_match else None

    # Try error patterns first
    for ep in ERROR_PATTERNS:
        m = ep["pattern"].search(line)
        if m:
            return {
                "timestamp": timestamp,
                "level": "ERROR",
                "category": ep["category"],
                "groups": m.groups(),
                "message": m.group(0),
                "engine_fix": ep["engine_fix"],
                "rust_area": ep["rust_area"],
                "raw": line.strip(),
            }

    # Try info patterns
    for ip in INFO_PATTERNS:
        m = ip["pattern"].search(line)
        if m:
            return {
                "timestamp": timestamp,
                "level": "INFO",
                "category": ip["category"],
                "groups": m.groups(),
                "message": m.group(0),
                "raw": line.strip(),
            }

    return None


def parse_log_file(log_path: Path) -> list[dict]:
    """Parse all RL-related entries from a log file."""
    if not log_path.exists():
        return []
    records = []
    with open(log_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            rec = parse_rl_log_line(line)
            if rec:
                records.append(rec)
    return records


# ---------------------------------------------------------------------------
# Deduplication and error registry
# ---------------------------------------------------------------------------

def error_key(record: dict) -> str:
    """Generate a dedup key for an error record."""
    # Key by category + core message (strip specific territory names for some categories)
    cat = record["category"]
    if cat == "impassable_territory":
        # Key: category + destination territory
        return f"{cat}::{record['groups'][1]}"
    elif cat == "no_transports":
        return f"{cat}::{record['groups'][0]}->{record['groups'][1]}"
    elif cat == "hostile_sea_loading":
        return f"{cat}::{record['groups'][1]}"
    elif cat == "chinese_restrictions":
        return f"{cat}::{record['groups'][0]}->{record['groups'][1]}"
    elif cat == "placement_invalid":
        return f"{cat}::{record['groups'][0]}::{record['groups'][1]}"
    else:
        return f"{cat}::{record['message']}"


def build_error_registry(records: list[dict]) -> dict:
    """Build a deduped error registry from parsed records."""
    registry: dict[str, dict] = {}

    # Load existing registry if present
    if ERROR_REGISTRY.exists():
        try:
            existing = json.loads(ERROR_REGISTRY.read_text())
            registry = {e["key"]: e for e in existing.get("errors", [])}
        except (json.JSONDecodeError, KeyError):
            pass

    for rec in records:
        if rec["level"] != "ERROR":
            continue
        key = error_key(rec)
        if key in registry:
            registry[key]["count"] += 1
            registry[key]["last_seen"] = datetime.now().isoformat()
        else:
            registry[key] = {
                "key": key,
                "category": rec["category"],
                "message": rec["message"],
                "engine_fix": rec.get("engine_fix", ""),
                "rust_area": rec.get("rust_area", ""),
                "count": 1,
                "first_seen": datetime.now().isoformat(),
                "last_seen": datetime.now().isoformat(),
                "status": "open",  # open | fixed | wontfix
            }

    return {
        "updated_at": datetime.now().isoformat(),
        "total_unique_errors": len(registry),
        "total_error_count": sum(e["count"] for e in registry.values()),
        "errors": sorted(registry.values(), key=lambda e: -e["count"]),
    }


def build_session_summary(records: list[dict]) -> dict:
    """Build a per-session summary with stats."""
    errors = [r for r in records if r["level"] == "ERROR"]
    infos = [r for r in records if r["level"] == "INFO"]

    # Count by category
    by_category = Counter(r["category"] for r in errors)

    # Extract phases seen
    phases_seen = []
    for r in infos:
        if r["category"] == "phase_received":
            phases_seen.append({
                "phase": r["groups"][0],
                "player": r["groups"][1],
                "round": int(r["groups"][2]),
                "pus": int(r["groups"][3]),
            })

    return {
        "session_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "timestamp": datetime.now().isoformat(),
        "total_records": len(records),
        "total_errors": len(errors),
        "total_info": len(infos),
        "errors_by_category": dict(by_category),
        "phases_seen": phases_seen,
        "deduped_errors": _dedup_errors_for_session(errors),
    }


def _dedup_errors_for_session(errors: list[dict]) -> list[dict]:
    """Deduplicate errors within a session for compact output."""
    seen: dict[str, dict] = {}
    for e in errors:
        key = error_key(e)
        if key in seen:
            seen[key]["count"] += 1
        else:
            seen[key] = {
                "key": key,
                "category": e["category"],
                "message": e["message"],
                "engine_fix": e.get("engine_fix", ""),
                "rust_area": e.get("rust_area", ""),
                "count": 1,
            }
    return sorted(seen.values(), key=lambda e: -e["count"])


# ---------------------------------------------------------------------------
# Output for auto-research consumption
# ---------------------------------------------------------------------------

def errors_for_research(registry_path: Path = ERROR_REGISTRY) -> str:
    """
    Return a compact, token-efficient string of open errors for LLM consumption.
    Deduped by key, sorted by count, with fix hints. Designed to be dropped into
    an auto-research prompt.
    """
    if not registry_path.exists():
        return "No error registry found. Run log_collector.py first."

    reg = json.loads(registry_path.read_text())
    open_errors = [e for e in reg["errors"] if e["status"] == "open"]

    if not open_errors:
        return "No open errors in registry."

    lines = [f"## Open RL Bot Errors ({len(open_errors)} unique, {sum(e['count'] for e in open_errors)} total occurrences)\n"]
    for e in open_errors:
        lines.append(f"- [{e['category']}] x{e['count']}: {e['message']}")
        lines.append(f"  Fix: {e['engine_fix']}")
        lines.append(f"  Rust area: {e['rust_area']}")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def collect_and_save():
    """One-shot: parse logs, save session + update registry."""
    GAME_LOGS_DIR.mkdir(parents=True, exist_ok=True)

    records = parse_log_file(TRIPLEA_LOG)
    if not records:
        print("No RL bot log entries found in", TRIPLEA_LOG)
        return

    # Save session
    session = build_session_summary(records)
    session_file = GAME_LOGS_DIR / f"{session['session_id']}_session.json"
    session_file.write_text(json.dumps(session, indent=2))
    print(f"Session saved: {session_file}")
    print(f"  {session['total_errors']} errors, {session['total_info']} info records")

    # Update registry
    registry = build_error_registry(records)
    ERROR_REGISTRY.write_text(json.dumps(registry, indent=2))
    print(f"Registry updated: {registry['total_unique_errors']} unique errors, "
          f"{registry['total_error_count']} total occurrences")

    return session, registry


def print_summary():
    """Print deduped error summary for quick review."""
    if not ERROR_REGISTRY.exists():
        print("No error registry. Run without --summary first.")
        return

    reg = json.loads(ERROR_REGISTRY.read_text())
    print(f"\n{'='*70}")
    print(f"RL Bot Error Registry — {reg['total_unique_errors']} unique errors, "
          f"{reg['total_error_count']} total")
    print(f"Updated: {reg['updated_at']}")
    print(f"{'='*70}\n")

    by_cat = defaultdict(list)
    for e in reg["errors"]:
        by_cat[e["category"]].append(e)

    for cat, errors in sorted(by_cat.items(), key=lambda x: -sum(e["count"] for e in x[1])):
        total = sum(e["count"] for e in errors)
        status_counts = Counter(e["status"] for e in errors)
        print(f"\n## {cat} ({total} occurrences, {len(errors)} unique)")
        print(f"   Status: {dict(status_counts)}")
        for e in errors[:5]:  # Show top 5 per category
            print(f"   [{e['status']}] x{e['count']}: {e['message']}")
        if len(errors) > 5:
            print(f"   ... and {len(errors)-5} more")


def main():
    parser = argparse.ArgumentParser(description="RL Bot Log Collector")
    parser.add_argument("--summary", action="store_true", help="Print deduped error summary")
    parser.add_argument("--research", action="store_true", help="Print compact output for auto-research")
    parser.add_argument("--watch", action="store_true", help="Continuously watch and parse logs")
    args = parser.parse_args()

    if args.summary:
        print_summary()
    elif args.research:
        print(errors_for_research())
    elif args.watch:
        print(f"Watching {TRIPLEA_LOG} for RL bot entries...")
        last_size = 0
        while True:
            try:
                if TRIPLEA_LOG.exists():
                    current_size = TRIPLEA_LOG.stat().st_size
                    if current_size != last_size:
                        collect_and_save()
                        last_size = current_size
                time.sleep(30)
            except KeyboardInterrupt:
                print("\nStopped watching.")
                break
    else:
        collect_and_save()


if __name__ == "__main__":
    main()
