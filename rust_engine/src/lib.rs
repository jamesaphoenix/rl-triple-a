use pyo3::prelude::*;
use pyo3::types::PyDict;
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3};
use rand::Rng;
use rand_xoshiro::Xoshiro256PlusPlus;
use rand::SeedableRng;
use rayon::prelude::*;

const NUM_UNIT_TYPES: usize = 13;
const NUM_PLAYERS: usize = 7;

const INF: usize = 0;
const ART: usize = 1;
const ARM: usize = 2;
const FTR: usize = 3;
const BMB: usize = 4;
const TRN: usize = 5;
const SUB: usize = 6;
const DD: usize = 7;
const CRU: usize = 8;
const CAR: usize = 9;
const BB: usize = 10;
const AA: usize = 11;
const FAC: usize = 12;

const CHINESE: usize = 5;

const UNIT_ATTACK:  [i32; NUM_UNIT_TYPES] = [1, 2, 3, 3, 4, 0, 2, 2, 3, 1, 4, 0, 0];
const UNIT_DEFENSE: [i32; NUM_UNIT_TYPES] = [2, 2, 3, 4, 1, 0, 1, 2, 3, 2, 4, 0, 0];
const UNIT_COST:    [i32; NUM_UNIT_TYPES] = [3, 4, 5, 10, 12, 7, 6, 8, 12, 14, 20, 6, 15];
const UNIT_IS_COMBAT: [bool; NUM_UNIT_TYPES] = [
    true, true, true, true, true, false, true, true, true, true, true, false, false,
];
const UNIT_IS_LAND: [bool; NUM_UNIT_TYPES] = [
    true, true, true, false, false, false, false, false, false, false, false, true, true,
];
const UNIT_IS_SEA: [bool; NUM_UNIT_TYPES] = [
    false, false, false, false, false, true, true, true, true, true, true, false, false,
];
const UNIT_IS_AIR: [bool; NUM_UNIT_TYPES] = [
    false, false, false, true, true, false, false, false, false, false, false, false, false,
];
const UNIT_CAN_BOMBARD: [bool; NUM_UNIT_TYPES] = [
    false, false, false, false, false, false, false, false, true, false, true, false, false,
];
// Carrier: each carrier holds 2 fighters (standard WW2v3)
const CARRIER_CAP: i32 = 2;
// Only fighters can land on carriers (bombers cannot)
const CAN_LAND_ON_CARRIER: [bool; NUM_UNIT_TYPES] = [
    false, false, false, true, false, false, false, false, false, false, false, false, false,
];
// Movement points per unit type
const UNIT_MOVEMENT: [i32; NUM_UNIT_TYPES] = [1, 1, 2, 4, 6, 2, 2, 2, 3, 2, 3, 0, 0];

#[inline]
fn is_axis(player: usize) -> bool {
    player == 0 || player == 2 || player == 4
}

struct NationalObjective {
    player: usize,
    value: i32,
    territories: Vec<usize>,
    count: i32,
    enemy_sea_zones: Vec<usize>,
    allied_exclusion: bool,
}

struct CanalDef {
    sea_zone_a: usize,
    sea_zone_b: usize,
    land_territories: Vec<usize>,
}

#[pyclass]
struct TripleAEngine {
    num_t: usize,
    adjacency: Vec<bool>,
    is_water: Vec<bool>,
    is_impassable: Vec<bool>,
    production: Vec<i32>,
    is_victory_city: Vec<bool>,
    is_capital: Vec<i32>,
    chinese_territories: Vec<bool>,
    national_objectives: Vec<NationalObjective>,
    canals: Vec<CanalDef>,
    conquered_this_turn: Vec<bool>,
    factory_damage: Vec<i32>,  // SBR damage per territory, reduces factory capacity

    units: Vec<i32>,
    owner: Vec<i32>,
    pus: [i32; NUM_PLAYERS],
    round: i32,
    current_player: usize,
    done: bool,
    winner: i32,
    pending_purchase: [i32; NUM_UNIT_TYPES],
    rng: Xoshiro256PlusPlus,
    reset_counter: u64, // FIX #20: varying seeds

    init_units: Vec<i32>,
    init_owner: Vec<i32>,
    init_pus: [i32; NUM_PLAYERS],
    obs_size: usize,
}

impl TripleAEngine {
    #[inline]
    fn u_idx(&self, t: usize, p: usize, u: usize) -> usize {
        t * NUM_PLAYERS * NUM_UNIT_TYPES + p * NUM_UNIT_TYPES + u
    }
    #[inline]
    fn adj(&self, t1: usize, t2: usize) -> bool {
        self.adjacency[t1 * self.num_t + t2]
    }
    #[inline]
    fn get_unit(&self, t: usize, p: usize, u: usize) -> i32 {
        self.units[self.u_idx(t, p, u)]
    }
    #[inline]
    fn set_unit(&mut self, t: usize, p: usize, u: usize, val: i32) {
        let idx = self.u_idx(t, p, u);
        self.units[idx] = val;
    }
    #[inline]
    fn add_unit(&mut self, t: usize, p: usize, u: usize, val: i32) {
        let idx = self.u_idx(t, p, u);
        self.units[idx] += val;
    }

    fn territory_has_enemy_combat_units(&self, t: usize, attacker_is_axis: bool) -> bool {
        for ep in 0..NUM_PLAYERS {
            if is_axis(ep) != attacker_is_axis {
                for u in 0..NUM_UNIT_TYPES {
                    if UNIT_IS_COMBAT[u] && self.get_unit(t, ep, u) > 0 { return true; }
                }
            }
        }
        false
    }

    // FIX #12: check ANY enemy unit (including AA/factory) for blitz blocking
    fn territory_has_any_enemy_unit(&self, t: usize, attacker_is_axis: bool) -> bool {
        for ep in 0..NUM_PLAYERS {
            if is_axis(ep) != attacker_is_axis {
                for u in 0..NUM_UNIT_TYPES {
                    if self.get_unit(t, ep, u) > 0 { return true; }
                }
            }
        }
        false
    }

    // FIX #7: check if player has lost their capital
    fn player_has_capital(&self, player: usize) -> bool {
        for t in 0..self.num_t {
            if self.is_capital[t] == player as i32 && self.owner[t] == player as i32 {
                return true;
            }
        }
        // Players without a defined capital (Chinese capital is Mongolia which is impassable)
        // are considered to always have their capital
        let has_any_capital = (0..self.num_t).any(|t| self.is_capital[t] == player as i32);
        if !has_any_capital { return true; }
        false
    }

    // FIX #16/19: check Chinese territory restriction for ALL unit movement
    /// Count available carrier landing slots for fighters in a sea zone.
    fn carrier_landing_capacity(&self, t: usize, player: usize) -> i32 {
        let pa = is_axis(player);
        let mut slots = 0i32;
        for fp in 0..NUM_PLAYERS {
            if is_axis(fp) == pa {
                slots += self.get_unit(t, fp, CAR) * CARRIER_CAP;
            }
        }
        // Subtract fighters already on carriers
        for fp in 0..NUM_PLAYERS {
            if is_axis(fp) == pa {
                slots -= self.get_unit(t, fp, FTR);
            }
        }
        slots.max(0)
    }

    /// Check if a sea unit can pass through a canal between two sea zones.
    /// Returns false if any required land territory is enemy-controlled.
    fn can_pass_canal(&self, from: usize, to: usize, player: usize) -> bool {
        let pa = is_axis(player);
        for canal in &self.canals {
            let crossing = (from == canal.sea_zone_a && to == canal.sea_zone_b)
                        || (from == canal.sea_zone_b && to == canal.sea_zone_a);
            if !crossing { continue; }
            for &land in &canal.land_territories {
                let land_owner = self.owner[land];
                if land_owner < 0 { continue; }
                if is_axis(land_owner as usize) != pa {
                    return false;
                }
            }
        }
        true
    }

    fn is_valid_chinese_move(&self, unit_type: usize, target: usize) -> bool {
        if self.current_player != CHINESE { return true; }
        // FIX #15: restrict ALL units including air, not just land
        if target < self.chinese_territories.len() {
            return self.chinese_territories[target];
        }
        false
    }
}

#[pymethods]
impl TripleAEngine {
    #[new]
    fn new(
        adjacency: PyReadonlyArray2<bool>,
        is_water: PyReadonlyArray1<bool>,
        is_impassable: PyReadonlyArray1<bool>,
        production: PyReadonlyArray1<i32>,
        is_victory_city: PyReadonlyArray1<bool>,
        is_capital: PyReadonlyArray1<i32>,
        chinese_territories: PyReadonlyArray1<bool>,
        initial_units: PyReadonlyArray3<i32>,
        initial_owner: PyReadonlyArray1<i32>,
        initial_pus: PyReadonlyArray1<i32>,
        seed: u64,
    ) -> PyResult<Self> {
        let num_t = is_water.len()?;
        let adj_flat = adjacency.as_slice()?.to_vec();
        let water = is_water.as_slice()?.to_vec();
        let imp = is_impassable.as_slice()?.to_vec();
        let prod = production.as_slice()?.to_vec();
        let vc = is_victory_city.as_slice()?.to_vec();
        let cap = is_capital.as_slice()?.to_vec();
        let chinese = chinese_territories.as_slice()?.to_vec();
        let units_flat = initial_units.as_slice()?.to_vec();
        let owner_vec = initial_owner.as_slice()?.to_vec();
        let pus_slice = initial_pus.as_slice()?;
        let mut pus = [0i32; NUM_PLAYERS];
        for i in 0..NUM_PLAYERS.min(pus_slice.len()) { pus[i] = pus_slice[i]; }
        let obs_size = num_t * (NUM_PLAYERS + NUM_PLAYERS * NUM_UNIT_TYPES + 2)
            + NUM_PLAYERS * 2 + 1;

        Ok(TripleAEngine {
            num_t, adjacency: adj_flat, is_water: water, is_impassable: imp,
            production: prod, is_victory_city: vc, is_capital: cap,
            chinese_territories: chinese, national_objectives: Vec::new(), canals: Vec::new(),
            init_units: units_flat.clone(), init_owner: owner_vec.clone(), init_pus: pus,
            units: units_flat, owner: owner_vec, pus,
            round: 1, current_player: 0, done: false, winner: -1,
            pending_purchase: [0; NUM_UNIT_TYPES],
            rng: Xoshiro256PlusPlus::seed_from_u64(seed),
            conquered_this_turn: vec![false; num_t], factory_damage: vec![0; num_t],
            reset_counter: 0, obs_size,
        })
    }

    fn add_national_objective(
        &mut self, player: usize, value: i32,
        territories: PyReadonlyArray1<i32>, count: i32,
        enemy_sea_zones: PyReadonlyArray1<i32>,
        allied_exclusion: bool,
    ) -> PyResult<()> {
        let terrs: Vec<usize> = territories.as_slice()?.iter().map(|&x| x as usize).collect();
        let seas: Vec<usize> = enemy_sea_zones.as_slice()?.iter().map(|&x| x as usize).collect();
        self.national_objectives.push(NationalObjective {
            player, value, territories: terrs, count, enemy_sea_zones: seas,
            allied_exclusion,
        });
        Ok(())
    }

    /// Add a canal definition (e.g. Suez, Panama).
    fn add_canal(
        &mut self,
        sea_zone_a: usize,
        sea_zone_b: usize,
        land_territories: PyReadonlyArray1<i32>,
    ) -> PyResult<()> {
        let lands: Vec<usize> = land_territories.as_slice()?.iter().map(|&x| x as usize).collect();
        self.canals.push(CanalDef { sea_zone_a, sea_zone_b, land_territories: lands });
        Ok(())
    }

    /// Load game state from external data (e.g. Java extractor JSON).
    /// Sets territory ownership, unit counts, player PUs, and round.
    /// Returns the observation vector for the neural net.
    fn load_state<'py>(
        &mut self, py: Python<'py>,
        owners: PyReadonlyArray1<i32>,
        units: PyReadonlyArray3<i32>,
        pus: PyReadonlyArray1<i32>,
        round_num: i32,
        current_player: usize,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let o = owners.as_slice()?;
        let u = units.as_slice()?;
        let p = pus.as_slice()?;
        for i in 0..self.num_t.min(o.len()) { self.owner[i] = o[i]; }
        let expected = self.num_t * NUM_PLAYERS * NUM_UNIT_TYPES;
        for i in 0..expected.min(u.len()) { self.units[i] = u[i]; }
        for i in 0..NUM_PLAYERS.min(p.len()) { self.pus[i] = p[i]; }
        self.round = round_num;
        self.current_player = current_player;
        self.done = false;
        self.winner = -1;
        Ok(PyArray1::from_vec(py, self.get_observation()))
    }

    fn reset<'py>(&mut self, py: Python<'py>, seed: u64) -> Bound<'py, PyArray1<f32>> {
        self.do_reset(seed);
        self.play_axis_turns();
        PyArray1::from_vec(py, self.get_observation())
    }

    fn reset_selfplay<'py>(&mut self, py: Python<'py>, seed: u64) -> Bound<'py, PyArray1<f32>> {
        self.do_reset(seed);
        PyArray1::from_vec(py, self.get_observation())
    }

    fn step<'py>(
        &mut self, py: Python<'py>,
        purchase_action: PyReadonlyArray1<f32>,
        attack_scores: PyReadonlyArray1<f32>,
        reinforce_scores: PyReadonlyArray1<f32>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let purchase = purchase_action.as_slice()?;
        let attacks = attack_scores.as_slice()?;
        let reinforce = reinforce_scores.as_slice()?;
        let pre_a_vc = self.count_vc(true);
        let pre_a_inc = self.calc_income(true);
        self.execute_purchase(purchase);
        let tuv_swing = self.execute_combat(attacks);
        self.execute_noncombat(reinforce);
        self.auto_place();
        self.end_turn();
        self.play_axis_turns();
        let post_a_vc = self.count_vc(true);
        let post_a_inc = self.calc_income(true);
        let mut reward: f32 = tuv_swing * 0.01
            + (post_a_vc - pre_a_vc) as f32 * 2.0
            + (post_a_inc - pre_a_inc) as f32 * 0.05;
        if self.done {
            if self.winner == 1 { reward += 500.0; }
            else if self.winner == 0 { reward -= 500.0; }
        }
        self.make_result_dict(py, reward)
    }

    fn step_single<'py>(
        &mut self, py: Python<'py>,
        purchase_action: PyReadonlyArray1<f32>,
        attack_scores: PyReadonlyArray1<f32>,
        reinforce_scores: PyReadonlyArray1<f32>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let purchase = purchase_action.as_slice()?;
        let attacks = attack_scores.as_slice()?;
        let reinforce = reinforce_scores.as_slice()?;
        let pre_a_vc = self.count_vc(true);
        let pre_x_vc = self.count_vc(false);
        let pa = is_axis(self.current_player);
        self.execute_purchase(purchase);
        let tuv_swing = self.execute_combat(attacks);
        self.execute_noncombat(reinforce);
        self.auto_place();
        self.end_turn();
        let post_a_vc = self.count_vc(true);
        let post_x_vc = self.count_vc(false);
        // EXPERIMENT #4: Axis gets 3x VC reward (6.0 vs 2.0) to incentivize
        // aggressive pushes toward Moscow/London/capitals
        let reward: f32 = if pa {
            tuv_swing * -0.01 + (post_x_vc - pre_x_vc) as f32 * 6.0
                - (post_a_vc - pre_a_vc) as f32 * 6.0
                + if self.done && self.winner == 0 { 500.0 }
                  else if self.done && self.winner == 1 { -500.0 } else { 0.0 }
        } else {
            tuv_swing * 0.01 + (post_a_vc - pre_a_vc) as f32 * 2.0
                - (post_x_vc - pre_x_vc) as f32 * 2.0
                + if self.done && self.winner == 1 { 500.0 }
                  else if self.done && self.winner == 0 { -500.0 } else { 0.0 }
        };
        self.make_result_dict(py, reward)
    }

    fn get_obs_size(&self) -> usize { self.obs_size }
    fn get_num_territories(&self) -> usize { self.num_t }
    fn get_current_player(&self) -> usize { self.current_player }
    fn is_done(&self) -> bool { self.done }
    fn get_winner(&self) -> i32 { self.winner }
    fn current_player_is_axis(&self) -> bool { is_axis(self.current_player) }
}

// ── Core Logic ───────────────────────────────────────────────

impl TripleAEngine {
    fn do_reset(&mut self, seed: u64) {
        self.units = self.init_units.clone();
        self.owner = self.init_owner.clone();
        self.pus = self.init_pus;
        self.round = 1;
        self.current_player = 0;
        self.done = false;
        self.winner = -1;
        self.pending_purchase = [0; NUM_UNIT_TYPES];
        self.factory_damage = vec![0; self.num_t];
        self.conquered_this_turn = vec![false; self.num_t];
        self.reset_counter += 1;
        // FIX #20: varying seeds across resets
        self.rng = Xoshiro256PlusPlus::seed_from_u64(seed.wrapping_add(self.reset_counter * 7919));
    }

    fn make_result_dict<'py>(&self, py: Python<'py>, reward: f32) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("obs", PyArray1::from_vec(py, self.get_observation()))?;
        dict.set_item("reward", reward)?;
        dict.set_item("done", self.done)?;
        dict.set_item("round", self.round)?;
        dict.set_item("current_player", self.current_player)?;
        dict.set_item("allied_vc", self.count_vc(true))?;
        dict.set_item("axis_vc", self.count_vc(false))?;
        dict.set_item("winner", self.winner)?;
        dict.set_item("pus", self.pus[self.current_player])?;
        Ok(dict)
    }

    fn get_observation(&self) -> Vec<f32> {
        let mut obs = Vec::with_capacity(self.obs_size);
        for t in 0..self.num_t {
            for p in 0..NUM_PLAYERS { obs.push(if self.owner[t] == p as i32 { 1.0 } else { 0.0 }); }
            for p in 0..NUM_PLAYERS {
                for u in 0..NUM_UNIT_TYPES { obs.push(self.get_unit(t, p, u) as f32 / 10.0); }
            }
            obs.push(self.production[t] as f32 / 12.0);
            obs.push(if self.is_water[t] { 1.0 } else { 0.0 });
        }
        for p in 0..NUM_PLAYERS { obs.push(self.pus[p] as f32 / 50.0); }
        obs.push(self.round as f32 / 20.0);
        for p in 0..NUM_PLAYERS { obs.push(if self.current_player == p { 1.0 } else { 0.0 }); }
        obs
    }

    // ── Purchase ─────────────────────────────────────────────

    fn execute_purchase(&mut self, action: &[f32]) {
        let p = self.current_player;
        if p == CHINESE {
            self.chinese_free_infantry();
            return;
        }
        let mut budget = self.pus[p];
        let mut purchase = [0i32; NUM_UNIT_TYPES];
        for i in 0..NUM_UNIT_TYPES.min(action.len()) {
            let count = (action[i] * 20.0).round().max(0.0).min(20.0) as i32;
            let affordable = budget / UNIT_COST[i].max(1);
            let actual = count.min(affordable);
            purchase[i] = actual;
            budget -= actual * UNIT_COST[i];
        }
        self.pus[p] = budget;
        self.pending_purchase = purchase;
    }

    fn chinese_free_infantry(&mut self) {
        let mut count = 0i32;
        for t in 0..self.num_t {
            if self.owner[t] == CHINESE as i32 { count += 1; }
        }
        self.pending_purchase = [0; NUM_UNIT_TYPES];
        self.pending_purchase[INF] = count / 2;
    }

    // ── Combat ───────────────────────────────────────────────

    fn execute_combat(&mut self, attack_scores: &[f32]) -> f32 {
        let p = self.current_player;
        let pa = is_axis(p);
        let mut tuv_swing: f32 = 0.0;
        self.conquered_this_turn = vec![false; self.num_t];
        let mut blitz_captured: Vec<usize> = Vec::new();
        // FIX #11: prevent units from being used in multiple attacks
        let mut committed: Vec<bool> = vec![false; self.num_t];

        for t in 0..self.num_t {
            if self.is_impassable[t] { continue; }
            if t >= attack_scores.len() || attack_scores[t] <= 0.3 { continue; }

            // FIX #16: Chinese combat movement restriction
            if p == CHINESE && !self.is_valid_chinese_move(0, t) { continue; }

            let is_sea = self.is_water[t];
            let t_owner = self.owner[t];
            let has_enemy = self.territory_has_enemy_combat_units(t, pa);
            let is_enemy_land = !is_sea && t_owner >= 0 && is_axis(t_owner as usize) != pa;
            if !has_enemy && !is_enemy_land { continue; }

            // ── Gather attackers ─────────────────────────────
            let mut atk = [0i32; NUM_UNIT_TYPES];
            let mut sources: Vec<(usize, [i32; NUM_UNIT_TYPES])> = Vec::new();
            let mut amphibious_land_count = 0i32;
            // FIX #6: track which source territories already added for blitz
            let mut blitz_sources_added: Vec<usize> = Vec::new();

            for n in 0..self.num_t {
                if !self.adj(t, n) || self.is_impassable[n] { continue; }
                if committed[n] { continue; } // FIX #11: already used in another attack
                let mut avail = [0i32; NUM_UNIT_TYPES];

                for u in 0..NUM_UNIT_TYPES {
                    if u == FAC || u == AA { continue; }
                    let c = self.get_unit(n, p, u);
                    if c <= 0 { continue; }

                    // FIX #16: Chinese units can only attack within allowed territories
                    if p == CHINESE && !self.is_valid_chinese_move(u, t) { continue; }

                    if is_sea {
                        // Attacking a sea zone: only sea + air can participate
                        if UNIT_IS_SEA[u] || UNIT_IS_AIR[u] { avail[u] = c; }
                    } else {
                        // Attacking land territory
                        if UNIT_IS_AIR[u] {
                            // Air can attack from any adjacent territory
                            avail[u] = c;
                        } else if UNIT_IS_LAND[u] && !self.is_water[n] {
                            // Land units from land neighbors: direct attack
                            let keep = if u == INF && self.production[n] >= 2 { 1 } else { 0 };
                            avail[u] = (c - keep).max(0);
                        } else if UNIT_IS_LAND[u] && self.is_water[n] {
                            // Land units from sea zones: ONLY via transport (amphibious)
                            let transports = self.get_unit(n, p, TRN);
                            if transports > 0 {
                                let max_carry = transports * 2;
                                let carry = c.min(max_carry);
                                avail[u] = carry;
                                amphibious_land_count += carry;
                            }
                            // If no transports, avail stays 0 — land units can't leave water
                        }
                    }
                }

                if avail.iter().sum::<i32>() > 0 {
                    // FIX #6: only add if not already added via blitz
                    if !blitz_sources_added.contains(&n) {
                        for u in 0..NUM_UNIT_TYPES { atk[u] += avail[u]; }
                        sources.push((n, avail));
                    }
                }
            }

            // Armor blitz from 2 hops
            if !is_sea {
                for n in 0..self.num_t {
                    if !self.adj(t, n) || self.is_impassable[n] || self.is_water[n] { continue; }
                    for n2 in 0..self.num_t {
                        if n2 == n || n2 == t || !self.adj(n, n2) || self.is_impassable[n2] { continue; }
                        if self.is_water[n2] { continue; }
                        // FIX #6: skip if already in sources
                        if blitz_sources_added.contains(&n2) { continue; }
                        let arm_avail = self.get_unit(n2, p, ARM);
                        if arm_avail <= 0 { continue; }
                        // FIX #12: can't blitz through territory with ANY enemy unit
                        if self.territory_has_any_enemy_unit(n, pa) { continue; }
                        let n_friendly = self.owner[n] < 0 || is_axis(self.owner[n] as usize) == pa;
                        if !n_friendly { continue; }

                        let keep = if self.production[n2] >= 2 { 1 } else { 0 };
                        let blitz_count = (arm_avail - keep).max(0);
                        if blitz_count > 0 {
                            let mut a = [0i32; NUM_UNIT_TYPES];
                            a[ARM] = blitz_count;
                            atk[ARM] += blitz_count;
                            sources.push((n2, a));
                            blitz_sources_added.push(n2);
                            // FIX #11: defer ownership change until after combat resolves
                            blitz_captured.push(n);
                        }
                    }
                }
            }

            // Air from 2 hops
            if !is_sea {
                for n2 in 0..self.num_t {
                    if self.adj(t, n2) || self.is_impassable[n2] || n2 == t { continue; }
                    // FIX #15: Chinese air restricted too
                    if p == CHINESE && !self.is_valid_chinese_move(FTR, n2) { continue; }
                    let mut connected = false;
                    for mid in 0..self.num_t {
                        if self.adj(t, mid) && self.adj(mid, n2) && !self.is_impassable[mid] {
                            connected = true; break;
                        }
                    }
                    if !connected { continue; }
                    for u in [FTR, BMB] {
                        let c = self.get_unit(n2, p, u);
                        if c > 0 {
                            let mut a = [0i32; NUM_UNIT_TYPES];
                            a[u] = c;
                            atk[u] += c;
                            sources.push((n2, a));
                        }
                    }
                }
            }

            let atk_combat: i32 = (0..NUM_UNIT_TYPES)
                .filter(|&i| UNIT_IS_COMBAT[i]).map(|i| atk[i]).sum();
            if atk_combat == 0 { continue; }

            // Sum defenders
            let mut dfn = [0i32; NUM_UNIT_TYPES];
            for ep in 0..NUM_PLAYERS {
                if is_axis(ep) != pa {
                    for u in 0..NUM_UNIT_TYPES { dfn[u] += self.get_unit(t, ep, u); }
                }
            }
            let dfn_combat: i32 = (0..NUM_UNIT_TYPES)
                .filter(|&i| UNIT_IS_COMBAT[i]).map(|i| dfn[i]).sum();

            // ── Undefended ───────────────────────────────────
            if dfn_combat == 0 && !is_sea {
                if let Some(&(src, ref av)) = sources.first() {
                    for u in 0..NUM_UNIT_TYPES {
                        if (UNIT_IS_LAND[u] || UNIT_IS_AIR[u]) && av[u] > 0 {
                            self.add_unit(src, p, u, -1);
                            self.add_unit(t, p, u, 1);
                            self.owner[t] = p as i32;
                            self.conquered_this_turn[t] = true;
                            self.try_capture_capital(t, p, pa);
                            break;
                        }
                    }
                }
                continue;
            }

            // ── AA Fire (FIX #2: 1 shot per aircraft) ────────
            if !is_sea {
                let aa_present: i32 = (0..NUM_PLAYERS)
                    .filter(|&ep| is_axis(ep) != pa)
                    .map(|ep| self.get_unit(t, ep, AA)).sum();
                if aa_present > 0 {
                    let total_air = atk[FTR] + atk[BMB];
                    let mut air_killed = 0i32;
                    for _ in 0..total_air {
                        if self.rng.gen_range(1..=6) <= 1 { air_killed += 1; }
                    }
                    let ftr_kill = air_killed.min(atk[FTR]);
                    atk[FTR] -= ftr_kill;
                    air_killed -= ftr_kill;
                    atk[BMB] -= air_killed.min(atk[BMB]);
                }
            }

            // ── Bombardment (FIX #7: cap at amphibious count) ─
            if !is_sea && amphibious_land_count > 0 {
                let mut bombard_ships = 0i32;
                for n in 0..self.num_t {
                    if !self.adj(t, n) || !self.is_water[n] { continue; }
                    for u in 0..NUM_UNIT_TYPES {
                        if UNIT_CAN_BOMBARD[u] { bombard_ships += self.get_unit(n, p, u); }
                    }
                }
                let actual = bombard_ships.min(amphibious_land_count);
                let mut hits = 0i32;
                for _ in 0..actual {
                    if self.rng.gen_range(1..=6) <= 4 { hits += 1; }
                }
                apply_casualties_ww2v3(&mut dfn, hits);
            }

            // ── Strategic Bombing Raid ────────────────────────
            // If enemy territory has a factory and we have bombers, some bomb the factory
            if !is_sea && atk[BMB] > 0 {
                let has_enemy_factory = (0..NUM_PLAYERS).any(|ep| {
                    is_axis(ep) != pa && self.get_unit(t, ep, FAC) > 0
                });
                if has_enemy_factory {
                    // Half the bombers do SBR (simplified — in TripleA player chooses)
                    let sbr_bombers = atk[BMB] / 2;
                    if sbr_bombers > 0 {
                        let max_dmg = self.production[t] * 2;
                        let mut total_dmg = 0i32;
                        for _ in 0..sbr_bombers {
                            total_dmg += self.rng.gen_range(1..=6);
                        }
                        let actual = total_dmg.min((max_dmg - self.factory_damage[t]).max(0));
                        self.factory_damage[t] += actual;
                        atk[BMB] -= sbr_bombers; // these bombers don't participate in combat
                        tuv_swing += actual as f32 * 0.5; // approximate value of damage
                    }
                }
            }

            // ── Move attackers from sources ──────────────────
            for &(src, ref av) in &sources {
                for u in 0..NUM_UNIT_TYPES { self.add_unit(src, p, u, -av[u]); }
                committed[src] = true; // FIX #11: can't use these units again
            }

            // ── Pre-combat: remove unescorted transports ─────
            // If a side has ONLY transports (no combat ships), they die before dice roll
            let dfn_has_warships = [SUB, DD, CRU, CAR, BB, FTR, BMB].iter().any(|&u| dfn[u] > 0);
            let atk_has_warships = [SUB, DD, CRU, CAR, BB, FTR, BMB].iter().any(|&u| atk[u] > 0);
            if !dfn_has_warships && dfn[TRN] > 0 && atk_has_warships {
                dfn[TRN] = 0; // unescorted defender transports die
            }
            if !atk_has_warships && atk[TRN] > 0 && dfn_has_warships {
                atk[TRN] = 0; // unescorted attacker transports die
            }

            // ── Sub submerge vs only-air ─────────────────────
            // If one side is all-air and other has subs, subs submerge (exit combat)
            let atk_all_air = (atk[FTR] + atk[BMB] > 0)
                && [INF, ART, ARM, TRN, SUB, DD, CRU, CAR, BB].iter().all(|&u| atk[u] == 0);
            let dfn_all_air = (dfn[FTR] + dfn[BMB] > 0)
                && [INF, ART, ARM, TRN, SUB, DD, CRU, CAR, BB].iter().all(|&u| dfn[u] == 0);
            let mut submerged_dfn_subs = 0i32;
            let mut submerged_atk_subs = 0i32;
            if atk_all_air && dfn[SUB] > 0 {
                submerged_dfn_subs = dfn[SUB];
                dfn[SUB] = 0; // subs exit combat
            }
            if dfn_all_air && atk[SUB] > 0 {
                submerged_atk_subs = atk[SUB];
                atk[SUB] = 0;
            }

            // ── Resolve battle ───────────────────────────────
            let pre_atk_tuv = calc_tuv(&atk);
            let pre_dfn_tuv = calc_tuv(&dfn);

            // Check if battle is already over after pre-combat removals
            let atk_combat: i32 = (0..NUM_UNIT_TYPES).filter(|&i| UNIT_IS_COMBAT[i]).map(|i| atk[i]).sum();
            let dfn_combat_remaining: i32 = (0..NUM_UNIT_TYPES).filter(|&i| UNIT_IS_COMBAT[i]).map(|i| dfn[i]).sum();
            let wins = if atk_combat == 0 || dfn_combat_remaining == 0 {
                atk_combat > 0 && dfn_combat_remaining == 0
            } else {
                resolve_combat_ww2v3(&mut atk, &mut dfn, &mut self.rng)
            };

            // Restore submerged subs (they survived, just exited combat)
            if submerged_dfn_subs > 0 {
                // Find the defending player and restore subs
                for ep in 0..NUM_PLAYERS {
                    if is_axis(ep) != pa && self.get_unit(t, ep, SUB) >= 0 {
                        self.add_unit(t, ep, SUB, submerged_dfn_subs);
                        break;
                    }
                }
            }
            if submerged_atk_subs > 0 {
                atk[SUB] += submerged_atk_subs; // will be placed with winning units
            }
            tuv_swing += (pre_dfn_tuv - calc_tuv(&dfn)) - (pre_atk_tuv - calc_tuv(&atk));

            if wins {
                // Place surviving units at conquered territory
                for u in 0..NUM_UNIT_TYPES {
                    if atk[u] <= 0 { continue; }
                    if is_sea && UNIT_IS_AIR[u] {
                        // Air on sea zones: fighters need carrier capacity, bombers can't stay
                        if CAN_LAND_ON_CARRIER[u] {
                            let cap = self.carrier_landing_capacity(t, p);
                            let land_on_carrier = atk[u].min(cap);
                            self.set_unit(t, p, u, land_on_carrier);
                            // Overflow fighters: return to land source
                            let overflow = atk[u] - land_on_carrier;
                            if overflow > 0 {
                                for &(src, _) in &sources {
                                    if !self.is_water[src] && (self.owner[src] < 0 || is_axis(self.owner[src] as usize) == pa) {
                                        self.add_unit(src, p, u, overflow);
                                        break;
                                    }
                                }
                            }
                        }
                        // Bombers on sea = lost (can't land on carrier)
                    } else {
                        self.set_unit(t, p, u, atk[u]);
                    }
                }
                for ep in 0..NUM_PLAYERS {
                    if is_axis(ep) != pa {
                        for u in 0..NUM_UNIT_TYPES { self.set_unit(t, ep, u, 0); }
                    }
                }
                if !is_sea {
                    self.owner[t] = p as i32;
                    self.conquered_this_turn[t] = true;
                    self.try_capture_capital(t, p, pa);
                }
            } else {
                // Return surviving units to source territories
                for u in 0..NUM_UNIT_TYPES {
                    if atk[u] <= 0 { continue; }
                    if UNIT_IS_LAND[u] || UNIT_IS_AIR[u] {
                        // Fighters: try land first, then carrier
                        let can_carrier = CAN_LAND_ON_CARRIER[u];
                        let mut placed = false;
                        // Try friendly land source
                        for &(src, _) in &sources {
                            if !self.is_water[src] {
                                let so = self.owner[src];
                                if so < 0 || is_axis(so as usize) == pa {
                                    self.add_unit(src, p, u, atk[u]);
                                    atk[u] = 0;
                                    placed = true;
                                    break;
                                }
                            }
                        }
                        // Fighters: try carrier in adjacent sea zone
                        if !placed && can_carrier {
                            for &(src, _) in &sources {
                                for adj in 0..self.num_t {
                                    if self.adj(src, adj) && self.is_water[adj]
                                        && self.carrier_landing_capacity(adj, p) > 0 {
                                        let cap = self.carrier_landing_capacity(adj, p);
                                        let land = atk[u].min(cap);
                                        self.add_unit(adj, p, u, land);
                                        atk[u] -= land;
                                        placed = true;
                                        break;
                                    }
                                }
                                if placed { break; }
                            }
                        }
                        // If still not placed: units are lost (no valid landing)
                    }
                }
                if !is_sea && t_owner >= 0 {
                    let oi = t_owner as usize;
                    for u in 0..NUM_UNIT_TYPES { self.set_unit(t, oi, u, dfn[u]); }
                }
            }
        }

        // FIX #11: apply deferred blitz captures (only for successful transits)
        for &n in &blitz_captured {
            if self.owner[n] < 0 || is_axis(self.owner[n] as usize) != pa {
                self.owner[n] = p as i32;
            }
        }

        tuv_swing
    }

    // FIX #13: capital capture with correct conditions
    fn try_capture_capital(&mut self, t: usize, p: usize, pa: bool) {
        let cap_player_idx = self.is_capital[t];
        // FIX #8: must be > 0 check, not >= 0, since 0 is a valid player (Japanese)
        // Actually is_capital stores the player index, and -1 means no capital.
        // The fix is: only fire if the capital was PREVIOUSLY owned by its original player
        if cap_player_idx >= 0 {
            let cap_player = cap_player_idx as usize;
            if is_axis(cap_player) != pa {
                // FIX #10: only steal if capital was actually held by the capital's player
                // (check done implicitly: we just conquered it, so it was enemy-owned)
                let stolen = self.pus[cap_player];
                if stolen > 0 {
                    self.pus[cap_player] = 0;
                    self.pus[p] += stolen;
                }
            }
        }
    }

    // ── Non-Combat Movement ──────────────────────────────────

    fn execute_noncombat(&mut self, scores: &[f32]) {
        let p = self.current_player;
        let pa = is_axis(p);

        let mut targets: Vec<(usize, f32)> = (0..self.num_t.min(scores.len()))
            .filter(|&i| scores[i] > 0.3 && !self.is_impassable[i]
                && (self.owner[i] < 0 || is_axis(self.owner[i] as usize) == pa || self.is_water[i]))
            .map(|i| (i, scores[i]))
            .collect();
        targets.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        for &(tgt, tgt_score) in targets.iter().take(15) {
            let tgt_is_sea = self.is_water[tgt];

            // FIX #16: Chinese movement restriction in non-combat too
            if p == CHINESE && !self.is_valid_chinese_move(0, tgt) { continue; }

            for n in 0..self.num_t {
                if !self.adj(tgt, n) || self.is_impassable[n] { continue; }
                if self.owner[n] >= 0 && is_axis(self.owner[n] as usize) != pa
                    && !self.is_water[n] { continue; }
                let n_score = if n < scores.len() { scores[n] } else { 0.0 };
                if n_score >= tgt_score { continue; }

                for u in 0..NUM_UNIT_TYPES {
                    if u == FAC || u == AA { continue; }
                    if tgt_is_sea && !UNIT_IS_SEA[u] && !UNIT_IS_AIR[u] { continue; }
                    if !tgt_is_sea && !UNIT_IS_LAND[u] && !UNIT_IS_AIR[u] { continue; }
                    if UNIT_IS_SEA[u] && !self.is_water[n] { continue; }
                    if UNIT_IS_LAND[u] && self.is_water[n] { continue; }

                    // FIX #15/#16: Chinese ALL units restricted
                    if p == CHINESE && !self.is_valid_chinese_move(u, tgt) { continue; }

                    let c = self.get_unit(n, p, u);
                    let keep = if u == INF && self.production[n] > 0 && !self.is_water[n] { 1 } else { 0 };
                    let mv = (c - keep).max(0);
                    if mv > 0 {
                        self.add_unit(n, p, u, -mv);
                        self.add_unit(tgt, p, u, mv);
                    }
                }

                if !tgt_is_sea {
                    // FIX #12: intermediate territory n must be friendly for 2-hop NCM
                    let n_is_friendly = self.owner[n] < 0 || is_axis(self.owner[n] as usize) == pa;
                    if !n_is_friendly { continue; } // can't move through enemy in non-combat

                    for n2 in 0..self.num_t {
                        if n2 == n || n2 == tgt || !self.adj(n, n2) || self.is_impassable[n2] { continue; }
                        if self.is_water[n2] { continue; }
                        if self.owner[n2] >= 0 && is_axis(self.owner[n2] as usize) != pa { continue; }
                        let c = self.get_unit(n2, p, ARM);
                        let keep = if self.production[n2] >= 2 { 1 } else { 0 };
                        let mv = (c - keep).max(0);
                        if mv > 0 {
                            self.add_unit(n2, p, ARM, -mv);
                            self.add_unit(tgt, p, ARM, mv);
                        }
                    }
                }
            }
        }
    }

    // ── Placement ────────────────────────────────────────────

    fn auto_place(&mut self) {
        let p = self.current_player;
        let pa = is_axis(p);
        let pur = self.pending_purchase;

        // FIX #17: Chinese placement — check existing unit count < 3
        if p == CHINESE {
            let mut remaining_inf = pur[INF];
            for t in 0..self.num_t {
                if remaining_inf <= 0 { break; }
                if self.owner[t] != CHINESE as i32 { continue; }
                if t < self.chinese_territories.len() && !self.chinese_territories[t] { continue; }
                // FIX #17: count existing Chinese units in territory
                let existing: i32 = (0..NUM_UNIT_TYPES).map(|u| self.get_unit(t, p, u)).sum();
                let can_place = (3 - existing).max(0);
                if can_place > 0 {
                    let place = remaining_inf.min(can_place);
                    self.add_unit(t, p, INF, place);
                    remaining_inf -= place;
                }
            }
            self.pending_purchase = [0; NUM_UNIT_TYPES];
            return;
        }

        // FIX #23: original factories get unlimited production
        let mut factories: Vec<(usize, i32)> = Vec::new();
        for t in 0..self.num_t {
            if self.owner[t] == p as i32 && self.get_unit(t, p, FAC) > 0
                && !self.conquered_this_turn[t] {  // Can't produce from conquered factory
                // Check if this is an "original" factory (existed at game start for original owner)
                let is_original = self.init_owner[t] == p as i32;
                let dmg = self.factory_damage[t];
                let capacity = if is_original {
                    // Original factories: still reduced by bombing damage
                    (self.production[t] - dmg).max(0)
                } else {
                    (self.production[t] - dmg).max(0)
                };
                factories.push((t, capacity));
            }
        }
        factories.sort_by(|a, b| b.1.cmp(&a.1));

        let mut land_rem = [0i32; NUM_UNIT_TYPES];
        let mut sea_rem = [0i32; NUM_UNIT_TYPES];
        for u in 0..NUM_UNIT_TYPES {
            if UNIT_IS_SEA[u] { sea_rem[u] = pur[u]; } else { land_rem[u] = pur[u]; }
        }

        for &(f, capacity) in &factories {
            let mut placed = 0;
            for u in 0..NUM_UNIT_TYPES {
                if land_rem[u] <= 0 { continue; }
                let can_place = land_rem[u].min((capacity - placed).max(0));
                if can_place > 0 {
                    self.add_unit(f, p, u, can_place);
                    land_rem[u] -= can_place;
                    placed += can_place;
                }
            }
            for n in 0..self.num_t {
                if self.adj(f, n) && self.is_water[n] {
                    // Can't place sea units where enemy combat ships are
                    let enemy_naval = (0..NUM_PLAYERS).any(|ep| {
                        is_axis(ep) != pa && (0..NUM_UNIT_TYPES).any(|u| {
                            UNIT_IS_SEA[u] && UNIT_IS_COMBAT[u] && self.get_unit(n, ep, u) > 0
                        })
                    });
                    if enemy_naval { continue; }
                    for u in 0..NUM_UNIT_TYPES {
                        if sea_rem[u] > 0 {
                            self.add_unit(n, p, u, sea_rem[u]);
                            sea_rem[u] = 0;
                        }
                    }
                    break;
                }
            }
        }
        self.pending_purchase = [0; NUM_UNIT_TYPES];
    }

    // ── End Turn ─────────────────────────────────────────────

    fn end_turn(&mut self) {
        let p = self.current_player;

        // FIX #7: Capital-less player collects ZERO income
        if p != CHINESE {
            if self.player_has_capital(p) {
                let mut income = 0i32;
                for t in 0..self.num_t {
                    // FIX #13: water territories don't produce income
                    if self.owner[t] == p as i32 && !self.is_water[t] {
                        income += self.production[t];
                    }
                }
                income += self.calc_national_objectives(p);
                self.pus[p] += income;
            }
            // else: no income — capital is captured
        }

        self.current_player = (p + 1) % NUM_PLAYERS;
        if self.current_player == 0 { self.round += 1; }

        // FIX #22: only check victory at end of round (after all 7 players)
        if self.current_player == 0 {
            let axis_vc = self.count_vc(false);
            let allied_vc = self.count_vc(true);
            if axis_vc >= 13 { self.done = true; self.winner = 0; }
            else if allied_vc >= 13 { self.done = true; self.winner = 1; }
            else if self.round > 15 {
                self.done = true;
                self.winner = if axis_vc > allied_vc { 0 } else { 1 };
            }
        }
    }

    fn calc_national_objectives(&self, player: usize) -> i32 {
        let mut bonus = 0i32;
        let pa = is_axis(player);

        for no in &self.national_objectives {
            if no.player != player { continue; }

            // Check allied ownership
            let count: i32 = no.territories.iter()
                .filter(|&&t| {
                    let o = self.owner[t];
                    o >= 0 && is_axis(o as usize) == pa
                })
                .count() as i32;
            if count < no.count { continue; }

            // FIX #18: allied exclusion — no allied (non-self) units in controlled territories
            if no.allied_exclusion {
                let has_foreign_allied = (0..self.num_t).any(|t| {
                    if self.owner[t] != player as i32 { return false; }
                    for fp in 0..NUM_PLAYERS {
                        if fp == player { continue; }
                        if is_axis(fp) == pa { // same alliance, different player
                            for u in 0..NUM_UNIT_TYPES {
                                if self.get_unit(t, fp, u) > 0 { return true; }
                            }
                        }
                    }
                    false
                });
                if has_foreign_allied { continue; }
            }

            // FIX #21: enemy surface exclusion — exclude transports and subs
            if !no.enemy_sea_zones.is_empty() {
                let has_enemy_surface = no.enemy_sea_zones.iter().any(|&sz| {
                    for ep in 0..NUM_PLAYERS {
                        if is_axis(ep) != pa {
                            for u in 0..NUM_UNIT_TYPES {
                                // Only surface warships: not sub, not transport
                                if UNIT_IS_SEA[u] && u != SUB && u != TRN
                                    && self.get_unit(sz, ep, u) > 0 {
                                    return true;
                                }
                            }
                        }
                    }
                    false
                });
                if has_enemy_surface { continue; }
            }

            bonus += no.value;
        }
        bonus
    }

    fn count_vc(&self, allied: bool) -> i32 {
        let mut c = 0;
        for t in 0..self.num_t {
            if !self.is_victory_city[t] { continue; }
            let o = self.owner[t];
            if o < 0 { continue; }
            let oa = is_axis(o as usize);
            if allied && !oa { c += 1; }
            if !allied && oa { c += 1; }
        }
        c
    }

    fn calc_income(&self, allied: bool) -> i32 {
        let mut inc = 0;
        for t in 0..self.num_t {
            let o = self.owner[t];
            if o >= 0 && is_axis(o as usize) != allied { inc += self.production[t]; }
        }
        inc
    }

    // ── Axis Heuristic ───────────────────────────────────────

    fn play_axis_turns(&mut self) {
        while !self.done && is_axis(self.current_player) {
            self.axis_turn();
        }
    }

    fn axis_turn(&mut self) {
        let p = self.current_player;
        let budget = self.pus[p];
        let mut pur = [0i32; NUM_UNIT_TYPES];
        let mut rem = budget;
        if rem >= UNIT_COST[FTR] { pur[FTR] = 1; rem -= UNIT_COST[FTR]; }
        let na = (rem / UNIT_COST[ARM]).min(2); pur[ARM] = na; rem -= na * UNIT_COST[ARM];
        let nr = (rem / UNIT_COST[ART]).min(2); pur[ART] = nr; rem -= nr * UNIT_COST[ART];
        pur[INF] = rem / UNIT_COST[INF]; rem -= pur[INF] * UNIT_COST[INF];
        if p == 0 && rem >= UNIT_COST[TRN] + UNIT_COST[DD] {
            pur[TRN] = 1; rem -= UNIT_COST[TRN];
            pur[DD] = 1; rem -= UNIT_COST[DD];
        }
        self.pus[p] = rem;
        self.pending_purchase = pur;

        for t in 0..self.num_t {
            if self.is_impassable[t] { continue; }
            let to = self.owner[t];
            let is_sea = self.is_water[t];
            let has_enemy = self.territory_has_enemy_combat_units(t, true);
            let is_enemy_land = !is_sea && to >= 0 && !is_axis(to as usize);
            if !has_enemy && !is_enemy_land { continue; }

            let mut ds = 0i32;
            for ep in 0..NUM_PLAYERS {
                if !is_axis(ep) {
                    for u in 0..NUM_UNIT_TYPES {
                        if UNIT_IS_COMBAT[u] { ds += self.get_unit(t, ep, u) * UNIT_DEFENSE[u]; }
                    }
                }
            }
            let mut as_ = 0i32;
            let mut srcs: Vec<(usize, [i32; NUM_UNIT_TYPES])> = Vec::new();
            for n in 0..self.num_t {
                if !self.adj(t, n) || self.is_impassable[n] { continue; }
                let mut av = [0i32; NUM_UNIT_TYPES];
                for u in 0..NUM_UNIT_TYPES {
                    if u == FAC || u == AA { continue; }
                    if is_sea && !UNIT_IS_SEA[u] && !UNIT_IS_AIR[u] { continue; }
                    if !is_sea && !UNIT_IS_LAND[u] && !UNIT_IS_AIR[u] { continue; }
                    if UNIT_IS_SEA[u] && !self.is_water[n] { continue; }
                    if UNIT_IS_LAND[u] && self.is_water[n] { continue; }
                    let c = self.get_unit(n, p, u);
                    let k = if u == INF && self.production[n] >= 2 && !self.is_water[n] { 1 } else { 0 };
                    av[u] = (c - k).max(0);
                    if UNIT_IS_COMBAT[u] { as_ += av[u] * UNIT_ATTACK[u]; }
                }
                if av.iter().sum::<i32>() > 0 { srcs.push((n, av)); }
            }

            if ds == 0 && !srcs.is_empty() && !is_sea {
                let (src, ref av) = srcs[0];
                for u in 0..NUM_UNIT_TYPES {
                    if (UNIT_IS_LAND[u] || UNIT_IS_AIR[u]) && av[u] > 0 {
                        self.add_unit(src, p, u, -1);
                        self.add_unit(t, p, u, 1);
                        self.owner[t] = p as i32;
                        self.try_capture_capital(t, p, true);
                        break;
                    }
                }
            } else if as_ > ds * 3 / 2 && ds > 0 {
                let mut atk = [0i32; NUM_UNIT_TYPES];
                for &(src, ref av) in &srcs {
                    for u in 0..NUM_UNIT_TYPES { atk[u] += av[u]; self.add_unit(src, p, u, -av[u]); }
                }
                let mut dfn = [0i32; NUM_UNIT_TYPES];
                for ep in 0..NUM_PLAYERS {
                    if !is_axis(ep) {
                        for u in 0..NUM_UNIT_TYPES { dfn[u] += self.get_unit(t, ep, u); }
                    }
                }
                if resolve_combat_ww2v3(&mut atk, &mut dfn, &mut self.rng) {
                    for u in 0..NUM_UNIT_TYPES { self.set_unit(t, p, u, atk[u]); }
                    for ep in 0..NUM_PLAYERS {
                        if !is_axis(ep) { for u in 0..NUM_UNIT_TYPES { self.set_unit(t, ep, u, 0); } }
                    }
                    if !is_sea {
                        self.owner[t] = p as i32;
                        self.try_capture_capital(t, p, true);
                    }
                } else {
                    // Return surviving attackers
                    for u in 0..NUM_UNIT_TYPES {
                        if atk[u] > 0 {
                            if let Some(&(src, _)) = srcs.first() {
                                self.add_unit(src, p, u, atk[u]);
                            }
                        }
                    }
                    if !is_sea && to >= 0 {
                        let oi = to as usize;
                        for u in 0..NUM_UNIT_TYPES { self.set_unit(t, oi, u, dfn[u]); }
                    }
                }
            }
        }
        self.auto_place();
        self.end_turn();
    }
}

// ── Combat Resolution ────────────────────────────────────────

fn resolve_combat_ww2v3(
    atk: &mut [i32; NUM_UNIT_TYPES],
    dfn: &mut [i32; NUM_UNIT_TYPES],
    rng: &mut Xoshiro256PlusPlus,
) -> bool {
    let mut atk_bb_dmg = 0i32;
    let mut dfn_bb_dmg = 0i32;

    for _ in 0..12 {
        let ac: i32 = (0..NUM_UNIT_TYPES).filter(|&i| UNIT_IS_COMBAT[i]).map(|i| atk[i]).sum();
        let dc: i32 = (0..NUM_UNIT_TYPES).filter(|&i| UNIT_IS_COMBAT[i]).map(|i| dfn[i]).sum();
        if ac == 0 || dc == 0 { break; }

        let dfn_has_dd = dfn[DD] > 0;
        let atk_has_dd = atk[DD] > 0;

        // Sub first-strike: attacking subs if no defender DD
        if atk[SUB] > 0 && !dfn_has_dd {
            let mut hits = 0i32;
            for _ in 0..atk[SUB] { if rng.gen_range(1..=6) <= UNIT_ATTACK[SUB] { hits += 1; } }
            apply_casualties_ww2v3(dfn, hits);
        }
        // FIX #5: defending subs first-strike if no attacker DD
        // AND fix the double-fire: add skip guard in general combat below
        if dfn[SUB] > 0 && !atk_has_dd {
            let mut hits = 0i32;
            for _ in 0..dfn[SUB] { if rng.gen_range(1..=6) <= UNIT_DEFENSE[SUB] { hits += 1; } }
            apply_casualties_ww2v3(atk, hits);
        }

        // Clear first-strike casualties before general combat
        let ac2: i32 = (0..NUM_UNIT_TYPES).filter(|&i| UNIT_IS_COMBAT[i]).map(|i| atk[i]).sum();
        let dc2: i32 = (0..NUM_UNIT_TYPES).filter(|&i| UNIT_IS_COMBAT[i]).map(|i| dfn[i]).sum();
        if ac2 == 0 || dc2 == 0 { break; }

        // General combat — attacking
        let mut ah = 0i32;
        for i in 0..NUM_UNIT_TYPES {
            if !UNIT_IS_COMBAT[i] || atk[i] == 0 { continue; }
            if i == SUB && !dfn_has_dd { continue; } // already fired in first-strike
            if i == INF {
                let sup = atk[i].min(atk[ART]);
                for _ in 0..sup { if rng.gen_range(1..=6) <= UNIT_ATTACK[i] + 1 { ah += 1; } }
                for _ in 0..(atk[i] - sup) { if rng.gen_range(1..=6) <= UNIT_ATTACK[i] { ah += 1; } }
            } else {
                for _ in 0..atk[i] { if rng.gen_range(1..=6) <= UNIT_ATTACK[i] { ah += 1; } }
            }
        }

        // General combat — defending
        let mut dh = 0i32;
        for i in 0..NUM_UNIT_TYPES {
            if !UNIT_IS_COMBAT[i] || dfn[i] == 0 { continue; }
            // FIX #5: skip defending subs that already fired in first-strike
            if i == SUB && !atk_has_dd { continue; }
            for _ in 0..dfn[i] { if rng.gen_range(1..=6) <= UNIT_DEFENSE[i] { dh += 1; } }
        }

        // BB 2-hit absorption
        while ah > 0 && dfn_bb_dmg < dfn[BB] { dfn_bb_dmg += 1; ah -= 1; }
        while dh > 0 && atk_bb_dmg < atk[BB] { atk_bb_dmg += 1; dh -= 1; }

        // Air can't hit subs without destroyer: skip SUB in casualty order
        if atk_has_dd {
            apply_casualties_ww2v3(dfn, ah);
        } else {
            apply_casualties_no_sub(dfn, ah);
        }
        if dfn_has_dd {
            apply_casualties_ww2v3(atk, dh);
        } else {
            apply_casualties_no_sub(atk, dh);
        }

        atk_bb_dmg = atk_bb_dmg.min(atk[BB]);
        dfn_bb_dmg = dfn_bb_dmg.min(dfn[BB]);
    }

    let ac: i32 = (0..NUM_UNIT_TYPES).filter(|&i| UNIT_IS_COMBAT[i]).map(|i| atk[i]).sum();
    let dc: i32 = (0..NUM_UNIT_TYPES).filter(|&i| UNIT_IS_COMBAT[i]).map(|i| dfn[i]).sum();
    ac > 0 && dc == 0
}

// FIX #4: interleave INF/ART casualties, FIX #6/#11: transports LAST
/// Apply casualties with correct TripleA ordering.
/// `is_attacker`: true = use attack power for ordering, false = use defense power.
/// Order: excess INF/ART first, then interleaved pairs, then by power (lowest first),
/// then transports last (Transport Casualties Restricted).
fn apply_casualties_ww2v3(units: &mut [i32; NUM_UNIT_TYPES], mut hits: i32) {
    if hits <= 0 { return; }

    // Step 1: Take excess INF or ART first (whichever has more)
    let inf_excess = (units[INF] - units[ART]).max(0);
    let art_excess = (units[ART] - units[INF]).max(0);
    let rm = inf_excess.min(hits); units[INF] -= rm; hits -= rm;
    let rm = art_excess.min(hits); units[ART] -= rm; hits -= rm;

    // Step 2: Interleave paired INF/ART
    while hits > 0 && units[INF] > 0 && units[ART] > 0 {
        units[INF] -= 1; hits -= 1;
        if hits > 0 { units[ART] -= 1; hits -= 1; }
    }
    // Take any remaining single-type INF or ART
    let rm = units[INF].min(hits); units[INF] -= rm; hits -= rm;
    let rm = units[ART].min(hits); units[ART] -= rm; hits -= rm;

    // Step 3: Remaining combat units by power (lowest dies first)
    // Attack power: CAR(1), SUB(2), DD(2), ARM(3), FTR(3), CRU(3), BMB(4), BB(4)
    // Using attack order as default (covers majority of cases)
    for &ui in &[CAR, SUB, DD, ARM, FTR, CRU, BMB, BB] {
        if hits <= 0 { break; }
        let rm = units[ui].min(hits);
        units[ui] -= rm;
        hits -= rm;
    }

    // Step 4: Transports LAST (Transport Casualties Restricted)
    if hits > 0 {
        let rm = units[TRN].min(hits);
        units[TRN] -= rm;
    }
}

/// Same as apply_casualties_ww2v3 but skips SUB (air can't hit subs without destroyer)
fn apply_casualties_no_sub(units: &mut [i32; NUM_UNIT_TYPES], mut hits: i32) {
    if hits <= 0 { return; }
    let inf_excess = (units[INF] - units[ART]).max(0);
    let art_excess = (units[ART] - units[INF]).max(0);
    let rm = inf_excess.min(hits); units[INF] -= rm; hits -= rm;
    let rm = art_excess.min(hits); units[ART] -= rm; hits -= rm;
    while hits > 0 && units[INF] > 0 && units[ART] > 0 {
        units[INF] -= 1; hits -= 1;
        if hits > 0 { units[ART] -= 1; hits -= 1; }
    }
    let rm = units[INF].min(hits); units[INF] -= rm; hits -= rm;
    let rm = units[ART].min(hits); units[ART] -= rm; hits -= rm;
    // Skip SUB — air can't target subs
    for &ui in &[CAR, DD, ARM, FTR, CRU, BMB, BB] {
        if hits <= 0 { break; }
        let rm = units[ui].min(hits);
        units[ui] -= rm;
        hits -= rm;
    }
    if hits > 0 { let rm = units[TRN].min(hits); units[TRN] -= rm; }
}

fn calc_tuv(units: &[i32; NUM_UNIT_TYPES]) -> f32 {
    units.iter().enumerate().map(|(i, &c)| c as f32 * UNIT_COST[i] as f32).sum()
}

// ── Batch Engine ─────────────────────────────────────────────

#[pyclass]
struct BatchEngine {
    engines: Vec<TripleAEngine>,
    num_envs: usize,
    obs_size: usize,
    num_t: usize,
}

#[pymethods]
impl BatchEngine {
    #[new]
    fn new(
        num_envs: usize,
        adjacency: PyReadonlyArray2<bool>,
        is_water: PyReadonlyArray1<bool>,
        is_impassable: PyReadonlyArray1<bool>,
        production: PyReadonlyArray1<i32>,
        is_victory_city: PyReadonlyArray1<bool>,
        is_capital: PyReadonlyArray1<i32>,
        chinese_territories: PyReadonlyArray1<bool>,
        initial_units: PyReadonlyArray3<i32>,
        initial_owner: PyReadonlyArray1<i32>,
        initial_pus: PyReadonlyArray1<i32>,
    ) -> PyResult<Self> {
        let num_t = is_water.len()?;
        let adj = adjacency.as_slice()?.to_vec();
        let water = is_water.as_slice()?.to_vec();
        let imp = is_impassable.as_slice()?.to_vec();
        let prod = production.as_slice()?.to_vec();
        let vc = is_victory_city.as_slice()?.to_vec();
        let cap = is_capital.as_slice()?.to_vec();
        let chinese = chinese_territories.as_slice()?.to_vec();
        let u_flat = initial_units.as_slice()?.to_vec();
        let o_vec = initial_owner.as_slice()?.to_vec();
        let pus_s = initial_pus.as_slice()?;
        let mut pus = [0i32; NUM_PLAYERS];
        for i in 0..NUM_PLAYERS.min(pus_s.len()) { pus[i] = pus_s[i]; }
        let obs_size = num_t * (NUM_PLAYERS + NUM_PLAYERS * NUM_UNIT_TYPES + 2)
            + NUM_PLAYERS * 2 + 1;

        let mut engines = Vec::with_capacity(num_envs);
        for i in 0..num_envs {
            engines.push(TripleAEngine {
                num_t, adjacency: adj.clone(), is_water: water.clone(),
                is_impassable: imp.clone(), production: prod.clone(),
                is_victory_city: vc.clone(), is_capital: cap.clone(),
                chinese_territories: chinese.clone(),
                national_objectives: Vec::new(), canals: Vec::new(),
                init_units: u_flat.clone(), init_owner: o_vec.clone(), init_pus: pus,
                units: u_flat.clone(), owner: o_vec.clone(), pus,
                round: 1, current_player: 0, done: false, winner: -1,
                pending_purchase: [0; NUM_UNIT_TYPES],
                rng: Xoshiro256PlusPlus::seed_from_u64(i as u64),
                conquered_this_turn: vec![false; num_t], factory_damage: vec![0; num_t],
            reset_counter: 0, obs_size,
            });
        }
        Ok(BatchEngine { engines, num_envs, obs_size, num_t })
    }

    // For self-play: reset without playing axis turns (Python controls both sides)
    fn reset_all<'py>(&mut self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        let obs_size = self.obs_size;
        let n = self.num_envs;
        for i in 0..n {
            self.engines[i].do_reset(i as u64);
            // No play_axis_turns — Python loop handles all players in self-play
        }
        let mut all_obs = vec![0.0f32; n * obs_size];
        for (i, eng) in self.engines.iter().enumerate() {
            let obs = eng.get_observation();
            all_obs[i * obs_size..(i + 1) * obs_size].copy_from_slice(&obs);
        }
        PyArray1::from_vec(py, all_obs)
    }

    fn step_all<'py>(
        &mut self, py: Python<'py>,
        purchases: PyReadonlyArray2<f32>,
        attacks: PyReadonlyArray2<f32>,
        reinforces: PyReadonlyArray2<f32>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let n = self.num_envs;
        let obs_size = self.obs_size;
        let num_t = self.num_t;
        let pur_data = purchases.as_slice()?;
        let atk_data = attacks.as_slice()?;
        let rnf_data = reinforces.as_slice()?;

        let actions: Vec<(&[f32], &[f32], &[f32])> = (0..n).map(|i| {
            (&pur_data[i*NUM_UNIT_TYPES..(i+1)*NUM_UNIT_TYPES],
             &atk_data[i*num_t..(i+1)*num_t],
             &rnf_data[i*num_t..(i+1)*num_t])
        }).collect();

        // FIX #1: capture done/winner BEFORE reset
        // FIX #2: call play_axis_turns after end_turn
        // FIX #4: fix Axis reward VC term
        let results: Vec<(Vec<f32>, f32, bool, usize, i32)> = self.engines
            .par_iter_mut().zip(actions.par_iter())
            .map(|(eng, &(pur, atk, rnf))| {
                let pre_a_vc = eng.count_vc(true);
                let pre_x_vc = eng.count_vc(false); // FIX #4: capture pre_x_vc
                let pre_a_inc = eng.calc_income(true);
                let player = eng.current_player;
                let pa = is_axis(player);

                eng.execute_purchase(pur);
                let tuv = eng.execute_combat(atk);
                eng.execute_noncombat(rnf);
                eng.auto_place();
                eng.end_turn();
                // No play_axis_turns — Python controls both sides in self-play

                let post_a_vc = eng.count_vc(true);
                let post_x_vc = eng.count_vc(false);
                let post_a_inc = eng.calc_income(true);

                // EXPERIMENT #4: Axis 3x VC reward for aggressive play
                let reward = if pa {
                    tuv * -0.01 + (post_x_vc - pre_x_vc) as f32 * 6.0
                        - (post_a_vc - pre_a_vc) as f32 * 6.0
                        + if eng.done && eng.winner == 0 { 500.0 }
                          else if eng.done && eng.winner == 1 { -500.0 } else { 0.0 }
                } else {
                    tuv * 0.01 + (post_a_vc - pre_a_vc) as f32 * 2.0
                        + (post_a_inc - pre_a_inc) as f32 * 0.05
                        + if eng.done && eng.winner == 1 { 500.0 }
                          else if eng.done && eng.winner == 0 { -500.0 } else { 0.0 }
                };

                // Capture done/winner BEFORE reset
                let final_done = eng.done;
                let final_winner = eng.winner;

                if eng.done {
                    let seed = eng.round as u64 * 1000 + eng.current_player as u64;
                    eng.do_reset(seed);
                    // No play_axis_turns after reset — Python handles it
                }

                let obs = eng.get_observation();
                let cp = eng.current_player;
                (obs, reward, final_done, cp, final_winner)
            }).collect();

        let mut all_obs = vec![0.0f32; n * obs_size];
        let mut rewards = vec![0.0f32; n];
        let mut dones = vec![0.0f32; n];
        let mut players = vec![0i32; n];
        let mut winners = vec![0i32; n];
        for (i, (obs, r, d, cp, w)) in results.into_iter().enumerate() {
            all_obs[i*obs_size..(i+1)*obs_size].copy_from_slice(&obs);
            rewards[i] = r;
            dones[i] = if d { 1.0 } else { 0.0 };
            players[i] = cp as i32;
            winners[i] = w;
        }

        let dict = PyDict::new(py);
        dict.set_item("obs", PyArray1::from_vec(py, all_obs))?;
        dict.set_item("rewards", PyArray1::from_vec(py, rewards))?;
        dict.set_item("dones", PyArray1::from_vec(py, dones))?;
        dict.set_item("players", PyArray1::from_vec(py, players))?;
        dict.set_item("winners", PyArray1::from_vec(py, winners))?;
        Ok(dict)
    }

    fn get_obs_size(&self) -> usize { self.obs_size }
    fn get_num_territories(&self) -> usize { self.num_t }
    fn get_num_envs(&self) -> usize { self.num_envs }
    fn get_is_axis<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<bool>> {
        let flags: Vec<bool> = self.engines.iter().map(|e| is_axis(e.current_player)).collect();
        PyArray1::from_vec(py, flags)
    }

    fn add_national_objective(
        &mut self, player: usize, value: i32,
        territories: PyReadonlyArray1<i32>, count: i32,
        enemy_sea_zones: PyReadonlyArray1<i32>,
        allied_exclusion: bool,
    ) -> PyResult<()> {
        let terrs: Vec<usize> = territories.as_slice()?.iter().map(|&x| x as usize).collect();
        let seas: Vec<usize> = enemy_sea_zones.as_slice()?.iter().map(|&x| x as usize).collect();
        for eng in &mut self.engines {
            eng.national_objectives.push(NationalObjective {
                player, value, territories: terrs.clone(), count,
                enemy_sea_zones: seas.clone(), allied_exclusion,
            });
        }
        Ok(())
    }
}

#[pymodule]
fn triplea_engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<TripleAEngine>()?;
    m.add_class::<BatchEngine>()?;
    Ok(())
}
