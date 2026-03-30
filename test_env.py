#!/usr/bin/env python3
"""Quick test: verify the environment loads and runs."""

import sys
sys.path.insert(0, ".")

import numpy as np
from src.map_parser import load_default_map
from src.game_state import GameState, ALL_PLAYERS, AXIS_PLAYERS, ALLIED_PLAYERS, PLAYER_INDEX
from src.units import UNIT_TYPE_INDEX, PURCHASABLE_UNITS, NUM_UNIT_TYPES
from src.env import TripleAEnv


def test_map_loading():
    print("=== Testing Map Loading ===")
    map_data = load_default_map()

    land = [t for t in map_data.territories.values() if not t.is_water]
    sea = [t for t in map_data.territories.values() if t.is_water]
    impassable = [t for t in map_data.territories.values() if t.is_impassable]

    print(f"Total territories: {len(map_data.territories)}")
    print(f"  Land: {len(land)}")
    print(f"  Sea zones: {len(sea)}")
    print(f"  Impassable: {len(impassable)}")
    print(f"Players: {list(map_data.players.keys())}")
    print(f"Turn order: {map_data.turn_order}")
    print()

    for p in map_data.players.values():
        print(f"  {p.name}: {p.alliance}, {p.starting_pus} PUs")

    print()
    print("Capitals:")
    for t in map_data.territories.values():
        if t.is_capital:
            print(f"  {t.name} -> {t.capital_of} (production: {t.production})")

    print()
    print("Victory cities:")
    for t in map_data.territories.values():
        if t.is_victory_city:
            print(f"  {t.name} (owner: {t.owner}, production: {t.production})")

    return True


def test_game_state():
    print("\n=== Testing Game State ===")
    state = GameState()
    print(f"Num territories: {state.num_territories}")
    print(f"Observation size: {state.observation_size}")
    print(f"Current player: {state.current_player}")
    print(f"Round: {state.round}")

    print("\nStarting PUs:")
    for p in ALL_PLAYERS:
        p_idx = PLAYER_INDEX[p]
        income = state.get_player_income(p)
        print(f"  {p}: {state.pus[p_idx]} PUs, income={income}/turn")

    print(f"\nAllied victory cities: {state.count_victory_cities('Allies')}")
    print(f"Axis victory cities: {state.count_victory_cities('Axis')}")

    # Check some starting positions
    print("\nGermany units:")
    germany_idx = state.territory_index["Germany"]
    for p in ALL_PLAYERS:
        units = state.get_units_in_territory(germany_idx, p)
        if units:
            print(f"  {p}: {units}")

    print("\nRussia units:")
    russia_idx = state.territory_index["Russia"]
    for p in ALL_PLAYERS:
        units = state.get_units_in_territory(russia_idx, p)
        if units:
            print(f"  {p}: {units}")

    obs = state.to_observation("Russians")
    print(f"\nObservation shape: {obs.shape}")
    print(f"Observation range: [{obs.min():.3f}, {obs.max():.3f}]")

    return True


def test_environment():
    print("\n=== Testing Environment ===")
    env = TripleAEnv(seed=42, max_rounds=5)
    obs, info = env.reset()
    print(f"Initial obs shape: {obs.shape}")
    print(f"Initial info: {info}")

    total_reward = 0
    steps = 0
    for _ in range(20):
        # Random purchase action
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1

        if steps % 5 == 0:
            print(f"  Step {steps}: reward={reward:.2f}, round={info['round']}, "
                  f"allied_vc={info.get('allied_vc', '?')}, "
                  f"axis_vc={info.get('axis_vc', '?')}")

        if terminated or truncated:
            print(f"\nGame ended at round {info['round']}!")
            print(f"Winner: {info.get('winner', 'unknown')}")
            print(f"Total reward: {total_reward:.2f}")
            break

    return True


def test_combat():
    print("\n=== Testing Combat ===")
    from src.combat import resolve_combat, estimate_battle_odds

    # 3 infantry + 1 artillery attacking 2 infantry
    atk = np.zeros(NUM_UNIT_TYPES, dtype=np.int32)
    dfn = np.zeros(NUM_UNIT_TYPES, dtype=np.int32)
    atk[UNIT_TYPE_INDEX["infantry"]] = 3
    atk[UNIT_TYPE_INDEX["artillery"]] = 1
    dfn[UNIT_TYPE_INDEX["infantry"]] = 2

    odds = estimate_battle_odds(atk, dfn, num_simulations=1000)
    print(f"3 inf + 1 art vs 2 inf:")
    print(f"  Attacker wins: {odds['attacker_win_pct']:.1%}")
    print(f"  Defender wins: {odds['defender_win_pct']:.1%}")
    print(f"  Avg attacker TUV loss: {odds['avg_attacker_tuv_loss']:.1f}")
    print(f"  Avg defender TUV loss: {odds['avg_defender_tuv_loss']:.1f}")

    # 2 fighters + 1 bomber vs 1 fighter + 2 infantry
    atk2 = np.zeros(NUM_UNIT_TYPES, dtype=np.int32)
    dfn2 = np.zeros(NUM_UNIT_TYPES, dtype=np.int32)
    atk2[UNIT_TYPE_INDEX["fighter"]] = 2
    atk2[UNIT_TYPE_INDEX["bomber"]] = 1
    dfn2[UNIT_TYPE_INDEX["fighter"]] = 1
    dfn2[UNIT_TYPE_INDEX["infantry"]] = 2

    odds2 = estimate_battle_odds(atk2, dfn2, num_simulations=1000)
    print(f"\n2 ftr + 1 bmb vs 1 ftr + 2 inf:")
    print(f"  Attacker wins: {odds2['attacker_win_pct']:.1%}")
    print(f"  Defender wins: {odds2['defender_win_pct']:.1%}")

    return True


if __name__ == "__main__":
    ok = True
    ok &= test_map_loading()
    ok &= test_game_state()
    ok &= test_combat()
    ok &= test_environment()

    if ok:
        print("\n" + "=" * 60)
        print("  All tests passed! Ready to train.")
        print("  Run: python train.py")
        print("=" * 60)
    else:
        print("\nSome tests failed!")
        sys.exit(1)
