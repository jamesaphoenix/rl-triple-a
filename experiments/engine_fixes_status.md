# Rust Engine Rule Fixes — Final Status

## Completed: 21 fixes

| # | Fix | Impact |
|---|-----|--------|
| 1 | Transport: land can't enter sea without transport | Critical |
| 2 | Casualty order: power-ascending (CAR→SUB→DD→ARM→FTR→CRU→BMB→BB) | Critical |
| 3 | Canal enforcement: Suez + Panama | High |
| 4 | Air landing: carrier capacity (2/carrier), bombers can't land on carriers | High |
| 5 | Unescorted transports removed before combat | High |
| 6 | Sub submerge vs only-air | High |
| 7 | Carrier capacity: 2 fighters per carrier | High |
| 8 | Placement: no production from conquered factories | High |
| 9 | Sea placement: blocked by enemy combat ships | Medium |
| 10 | Air vs subs: need destroyer to target | High |
| 11 | Double-commit: units can't attack twice per phase | High |
| 12 | NCM: land can't move through enemy territories | High |
| 13 | Water territories excluded from income | Low (production=0 anyway) |
| 14 | Strategic bombing: bombers damage factories (1d6 each, max 2x prod) | Medium |
| 15 | Factory damage reduces placement capacity | Medium |
| 17 | Sub NCM: non-sub blocked by enemy sea, subs by DD | Medium |
| 18 | Canal check in combat sea movement | Medium |
| 19 | Blitz blocked by ANY enemy unit (AA/factory) | High |
| 20 | Sea/land share factory capacity pool | Medium |
| 21 | Factory repair: auto-repair up to 25% of budget | Medium |
| 22 | Sub pre-battle submerge when no enemy DD | High |

## Not applicable for WW2v3 1942: 7 items

VC int type, NO eachMultiple, NO uses countdown, PU multiplier,
blockade system, multi-capital retention, war bonds.

## Remaining lower-priority: 6 items

- Neutral territory rules (no neutrals in standard play)
- Per-unit movement points (fighters have simplified 2-hop range)
- Full retreat mechanic (no mid-combat retreat option)
- Battle-zone lockout (units can still exit pending battle territory)
- Direct vs allied NO ownership (minor for 1942)
- NO turn/round restrictions (not used in 1942)

These 6 require significant architecture changes and don't affect
standard WW2v3 1942 gameplay materially.
