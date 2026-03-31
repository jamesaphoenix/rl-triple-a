# Observation Compression Plan

## Problem
Current observation: 16,215 dimensions. First neural net layer: 8.3M of 8.9M total params.
93% of the network's capacity is spent projecting a massive input — most of which is noise
(empty territories, distant sea zones, impassable territories).

## Current Encoding (100 features per territory × 162 territories + 15 global)

Per territory (100 dims):
- Owner one-hot: 7
- Units per player per type: 7 × 13 = 91
- Production: 1
- Is water: 1

Global (15 dims):
- PUs per player: 7
- Round: 1
- Current player one-hot: 7

## Proposed Compressed Encoding (~2,500 dims)

### Step 1: Remove impassable territories (save ~1,500 dims)
15 territories are impassable (Afghanistan, Angola, Argentina, Eire, Himalaya, Mongolia,
Mozambique, Northern South America, Peruvian Central, Sahara, Saudi Arabia, Spain,
Sweden, Switzerland, Turkey). Remove them entirely.

**Remaining: 147 territories × 100 = 14,700 dims**

### Step 2: Compress unit encoding (save ~10,000 dims)
Instead of encoding ALL 7 players' units separately (91 dims per territory), encode:
- **Friendly units** (sum across alliance): 13 unit types = 13 dims
- **Enemy units** (sum across enemy alliance): 13 unit types = 13 dims
- **Total friendly strength**: 1 dim (weighted sum: atk×count)
- **Total enemy strength**: 1 dim (weighted sum: def×count)

Per territory: 7 (owner) + 13 (friendly units) + 13 (enemy units) + 2 (strength) + 1 (production) + 1 (is_water) = **37 dims**

**147 territories × 37 = 5,439 dims**

### Step 3: Add strategic features (add ~200 dims)
Per territory:
- Distance to nearest enemy capital: 1 dim
- Distance to own nearest capital: 1 dim
- Is frontline (adjacent to enemy): 1 dim
- Is victory city: 1 dim
- Has factory: 1 dim
- Adjacent friendly strength: 1 dim
- Adjacent enemy strength: 1 dim

Per territory: 37 + 7 = **44 dims**

**147 × 44 = 6,468 + 15 global = 6,483 dims**

### Step 4 (aggressive): Only encode relevant territories (~2,500 dims)
Only include territories that are:
- Owned by current player or adjacent to player's territories
- Within 2 hops of any frontline territory
- Victory cities
- Capitals
- Territories with factories

This typically covers ~60 territories. **60 × 44 = 2,640 + 15 = 2,655 dims**

## Implementation

### Rust changes (get_observation)
```rust
fn get_observation_compressed(&self) -> Vec<f32> {
    // 1. Skip impassable territories
    // 2. Encode friendly/enemy unit sums instead of per-player
    // 3. Add strategic features (frontline, distance)
    // 4. Global features unchanged
}
```

### Python changes
- `ActorCriticV2.__init__`: change `obs_size` parameter
- `train_phase2.py`: update obs_size
- `hud/server.py`: update obs_size

### Action space stays the same (337 dims)
Purchase (13) + attack scores (162) + reinforce scores (162)
Even with compressed obs, we still need to output scores for all territories.

## Expected Impact
- First layer: 8.3M params → ~1.3M params (84% reduction)
- Remaining layers unchanged (260k + 131k + policy/value heads)
- Total params: 8.9M → ~2M
- Signal-to-noise ratio: dramatically improved
- Training speed: faster forward/backward pass
- Expected effect: break the 81% ceiling within 200-300 iterations
