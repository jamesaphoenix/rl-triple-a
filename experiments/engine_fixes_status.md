# Rust Engine Rule Fixes Status

## Completed (19 fixes)

| # | Fix | Commit |
|---|-----|--------|
| 1 | Transport: land can't enter sea without transport | c4bf9a5 |
| 2 | Casualty order: power-ascending (CAR first, BB last) | c4bf9a5 |
| 3 | Canal enforcement: Suez + Panama | c4bf9a5 |
| 4 | Air landing: carrier capacity (2/carrier), bomber distinction | e3356bb |
| 5 | Unescorted transports: removed before combat | e3356bb |
| 6 | Sub submerge: exit combat vs only-air | e3356bb |
| 7 | Carrier capacity: 2 fighters per carrier | e3356bb |
| 8 | Placement: no production from conquered factories | 5aac4a0 |
| 9 | Sea placement: blocked by enemy combat ships | 5aac4a0 |
| 10 | Air vs subs: need destroyer to target | 82578cf |
| 11 | Double-commit: units can't attack twice per phase | b6a2f2b |
| 12 | NCM: land can't move through enemy territories (2-hop) | b6a2f2b |
| 13 | Water territories excluded from income | b6a2f2b |
| 14 | Strategic bombing: bombers damage factories (1d6 each) | 565ad73 |
| 15 | Factory damage reduces placement capacity | 565ad73 |
| 17 | Sub NCM: non-sub blocked by enemy sea, subs by DD | c4d2eaf |
| 18 | Canal check in combat sea movement | c4d2eaf |
| 19 | Blitz blocked by ANY enemy unit (AA/factory) | verified |
| 20 | Sea/land share factory capacity pool | c4d2eaf |

## Not applicable for WW2v3 1942 (7 items)

| # | Rule | Why N/A |
|---|------|---------|
| 16 | VC type bool→int | All 1942 VCs have value=1 |
| 24 | NO eachMultiple | Not used in 1942 |
| 25 | NO uses countdown | All 1942 NOs unlimited |
| 27 | PU multiplier | Default 1 in 1942 |
| 28 | Blockade system | No blockade zones in 1942 |
| 29 | Multi-capital retention | All players single capital |
| 34 | War bonds | Tech-gated, tech disabled |

## Remaining (8 items — lower priority)

| # | Rule | Complexity | Impact |
|---|------|-----------|--------|
| 23 | Direct vs allied NO ownership | Low | Minor for 1942 |
| 26 | NO turn/round restrictions | Low | Not used in 1942 |
| 30 | Neutral territory rules | Medium | No neutrals in standard play |
| 31 | Per-unit movement points | High | Fighters unlimited range (simplified) |
| 32 | Full retreat mechanic | High | No mid-combat retreat |
| 33 | Battle-zone lockout | Medium | Can move out of pending battle |
| 35 | Repair during purchase phase | Medium | Can't repair factory damage |
| 36 | Submarine retreat/submerge before battle | Medium | Sub retreat absent |
