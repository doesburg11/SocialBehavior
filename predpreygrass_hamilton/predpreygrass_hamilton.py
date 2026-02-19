#!/usr/bin/env python3
"""
predpreygrass_hamilton.py

Predator-prey-grass simulation focused on kin selection.

Key modeling choices:
- No cooperative hunting trait.
- Hunting is solo for predators.
- Cooperation is only kin-directed energy transfer using Hamilton's rule.
- Sexual demographics are explicit for both predators and prey.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# CONFIG
# ============================================================

# World
W, H = 60, 60
STEPS = 2500
SEED = 42
REPORT_EVERY = 200
RESTART_ON_EXTINCTION = False
MAX_RESTARTS = 30

# Population initialization
PRED_INIT = 35
PREY_INIT = 1000
PRED_ENERGY_INIT = 2.2
PREY_ENERGY_MEAN = 1.1
PREY_ENERGY_SIGMA = 0.25
PREY_ENERGY_MIN = 0.10

# Demographic constraints (ticks interpreted as years)
PRED_LIFE_EXPECTANCY = 75
PREY_LIFE_EXPECTANCY = 5

PRED_DEPENDENT_AGE_MAX = 16
PREY_DEPENDENT_AGE_MAX = 1

PRED_FEMALE_FERTILE_MIN = 16
PRED_FEMALE_FERTILE_MAX = 40
PREY_FEMALE_FERTILE_MIN = 2
PREY_FEMALE_FERTILE_MAX = 5

PRED_MALE_MATURE_AGE = 16
PREY_MALE_MATURE_AGE = 2

# Predator energetics / movement / solo hunting
PRED_METAB = 0.04
PRED_MOVE_PROB = 0.9
PRED_MOVE_COST = 0.010
PRED_HUNT_R = 0
PRED_SOLO_HUNT_SCALE = 0.05
PRED_HUNT_YIELD = 0.85

# Predator sexual reproduction
PRED_BIRTH_THRESH = 1.5
PRED_REPRO_PROB = 0.20
PRED_CHILD_SHARE = 0.25
PRED_FATHER_BIRTH_COST = 0.05
PRED_MALE_MIN_ENERGY = 0.5
PRED_MATE_R = 2
PRED_MAX = 900

# Prey energetics / movement / feeding
PREY_METAB = 0.045
PREY_MOVE_PROB = 0.30
PREY_MOVE_COST = 0.01
PREY_BITE_SIZE = 0.36

# Prey sexual reproduction
PREY_BIRTH_THRESH = 0.50
PREY_REPRO_PROB = 0.95
PREY_CHILD_SHARE = 0.45
PREY_FATHER_BIRTH_COST = 0.0
PREY_MALE_MIN_ENERGY = 0.60
PREY_MATE_R = 4
PREY_MAX = 5000

# Grass
GRASS_INIT = 0.8
GRASS_MAX = 3.0
GRASS_REGROWTH = 0.08

# Kin-selection transfer (Hamilton) - predators
PRED_KIN_TRANSFER_R = 6
PRED_KIN_TRANSFER_CHUNK = 0.04
PRED_KIN_MAX_CHUNKS_PER_DONOR = 4
PRED_KIN_RESERVE = 0.35
PRED_RECIP_SURV_SETPOINT = 2.0
PRED_RECIP_SURV_SCALE = 0.60
PRED_DONOR_FIT_SETPOINT = 1.2
PRED_DONOR_FIT_SCALE = 1.20

# Kin-selection transfer (Hamilton) - prey
PREY_KIN_TRANSFER_R = 1
PREY_KIN_TRANSFER_CHUNK = 0.03
PREY_KIN_MAX_CHUNKS_PER_DONOR = 4
PREY_KIN_RESERVE = 0.35
PREY_RECIP_SURV_SETPOINT = 0.6
PREY_RECIP_SURV_SCALE = 0.20
PREY_DONOR_FIT_SETPOINT = 0.95
PREY_DONOR_FIT_SCALE = 0.24

# If dependent kin are available, transfer targets them first.
DEPENDENT_PRIORITY = True

# Cue-based kin recognition model (decision-side r_hat)
KIN_CUE_MODEL = "maternal_residence"
KIN_CONSERVATIVE_BIAS = -0.15
KIN_W_MATERNAL = 0.75
KIN_W_CORESIDENCE = 0.35
MATERNAL_CUE_TPR = 0.85
MATERNAL_CUE_FPR = 0.005
CORESIDENCE_RADIUS_PRED = 2
CORESIDENCE_RADIUS_PREY = 1
CORESIDENCE_DECAY = 0.95
CORESIDENCE_INCREMENT = 1.0
CORESIDENCE_SATURATION = 8.0
TRUE_KIN_THRESHOLD = 0.25
EST_KIN_THRESHOLD = 0.25

# Founder kin seeding (set to 1 => unrelated founders)
PRED_FOUNDER_CLAN_SIZE = 1
PREY_FOUNDER_CLAN_SIZE = 1
FOUNDER_CLAN_R = 0.25

# Plotting
PLOT_RESULTS = True


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class Predator:
    aid: int
    x: int
    y: int
    energy: float
    sex: str  # "F" or "M"
    age: int
    mother_id: int | None = None
    father_id: int | None = None


@dataclass
class Prey:
    aid: int
    x: int
    y: int
    energy: float
    sex: str  # "F" or "M"
    age: int
    mother_id: int | None = None
    father_id: int | None = None


def wrap(v: int, L: int) -> int:
    return v % L


def random_sex() -> str:
    return "F" if random.random() < 0.5 else "M"


def sample_prey_energy() -> float:
    e = PREY_ENERGY_MEAN + random.gauss(0.0, PREY_ENERGY_SIGMA)
    return max(PREY_ENERGY_MIN, e)


def init_grass() -> np.ndarray:
    return np.full((H, W), GRASS_INIT, dtype=float)


def drain_energy(energy: float, amount: float) -> Tuple[float, float]:
    if amount <= 0.0 or energy <= 0.0:
        return energy, 0.0
    spent = min(energy, amount)
    return energy - spent, spent


def chebyshev_torus_distance(x1: int, y1: int, x2: int, y2: int) -> int:
    dx = abs(x1 - x2)
    dy = abs(y1 - y2)
    dx = min(dx, W - dx)
    dy = min(dy, H - dy)
    return max(dx, dy)


def build_spatial_index(agents: List[Predator | Prey]) -> Dict[Tuple[int, int], List[int]]:
    index: Dict[Tuple[int, int], List[int]] = {}
    for i, ag in enumerate(agents):
        index.setdefault((ag.x, ag.y), []).append(i)
    return index


def nearby_indices(
    index: Dict[Tuple[int, int], List[int]],
    x: int,
    y: int,
    radius: int,
) -> List[int]:
    out: List[int] = []
    for dy in range(-radius, radius + 1):
        yy = (y + dy) % H
        for dx in range(-radius, radius + 1):
            xx = (x + dx) % W
            out.extend(index.get((xx, yy), []))
    return out


# ============================================================
# PEDIGREE / RELATEDNESS
# ============================================================

def init_relatedness_matrix(ids: List[int]) -> Dict[int, Dict[int, float]]:
    rel: Dict[int, Dict[int, float]] = {}
    for aid in ids:
        rel[aid] = {aid: 1.0}
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            rel[a][b] = 0.0
            rel[b][a] = 0.0
    return rel


def coefficient_of_relationship(aid_a: int, aid_b: int, rel: Dict[int, Dict[int, float]]) -> float:
    if aid_a == aid_b:
        return 1.0
    return rel.get(aid_a, {}).get(aid_b, 0.0)


def add_child_relatedness(
    rel: Dict[int, Dict[int, float]],
    child_id: int,
    mother_id: int,
    father_id: int,
    existing_ids: List[int],
) -> None:
    rel.setdefault(child_id, {})
    rel[child_id][child_id] = 1.0

    mother_row = rel.get(mother_id, {})
    father_row = rel.get(father_id, {})

    for oid in existing_ids:
        if oid == child_id:
            continue

        r_mother_other = 1.0 if oid == mother_id else mother_row.get(oid, 0.0)
        r_father_other = 1.0 if oid == father_id else father_row.get(oid, 0.0)

        # Two-parent approximation for relationship propagation.
        r_child_other = 0.5 * (r_mother_other + r_father_other)

        rel[child_id][oid] = r_child_other
        rel.setdefault(oid, {oid: 1.0})
        rel[oid][child_id] = r_child_other


def prune_relatedness(rel: Dict[int, Dict[int, float]], alive_ids: Set[int]) -> None:
    stale_rows = [aid for aid in rel if aid not in alive_ids]
    for aid in stale_rows:
        rel.pop(aid, None)

    for aid, row in rel.items():
        stale_cols = [oid for oid in row if oid not in alive_ids]
        for oid in stale_cols:
            row.pop(oid, None)
        row[aid] = 1.0


def init_cue_memory(ids: List[int]) -> Dict[int, Dict[int, float]]:
    return {aid: {} for aid in ids}


def decay_cue_memory(memory: Dict[int, Dict[int, float]], decay: float, alive_ids: Set[int]) -> None:
    stale_rows = [aid for aid in memory if aid not in alive_ids]
    for aid in stale_rows:
        memory.pop(aid, None)

    for aid in alive_ids:
        memory.setdefault(aid, {})

    for aid, row in memory.items():
        stale_cols = [oid for oid in row if oid not in alive_ids]
        for oid in stale_cols:
            row.pop(oid, None)

        for oid in list(row.keys()):
            row[oid] *= decay
            if row[oid] < 1e-9:
                row.pop(oid, None)


def update_co_residence_memory(
    agents: List[Predator | Prey],
    memory: Dict[int, Dict[int, float]],
    radius: int,
    increment: float,
) -> None:
    if radius < 0 or increment <= 0.0:
        return

    for ag in agents:
        memory.setdefault(ag.aid, {})

    index = build_spatial_index(agents)
    for ag in agents:
        donor_id = ag.aid
        for idx in nearby_indices(index, ag.x, ag.y, radius):
            other = agents[idx]
            if other.aid == donor_id:
                continue
            row = memory.setdefault(donor_id, {})
            row[other.aid] = row.get(other.aid, 0.0) + increment
            row_other = memory.setdefault(other.aid, {})
            row_other[donor_id] = row[other.aid]


def prune_cue_memory(memory: Dict[int, Dict[int, float]], alive_ids: Set[int]) -> None:
    stale_rows = [aid for aid in memory if aid not in alive_ids]
    for aid in stale_rows:
        memory.pop(aid, None)

    for aid, row in memory.items():
        stale_cols = [oid for oid in row if oid not in alive_ids]
        for oid in stale_cols:
            row.pop(oid, None)


def seed_founder_clans(
    rel: Dict[int, Dict[int, float]],
    ids: List[int],
    clan_size: int,
    within_clan_r: float,
) -> None:
    if clan_size <= 1 or within_clan_r <= 0.0:
        return

    for start in range(0, len(ids), clan_size):
        clan = ids[start:start + clan_size]
        for i, aid_i in enumerate(clan):
            for aid_j in clan[i + 1:]:
                rel[aid_i][aid_j] = max(rel[aid_i].get(aid_j, 0.0), within_clan_r)
                rel[aid_j][aid_i] = rel[aid_i][aid_j]


# ============================================================
# HAMILTON TRANSFER MECHANICS
# ============================================================

def logistic_proxy(energy: float, setpoint: float, scale: float) -> float:
    s = max(1e-9, scale)
    z = (energy - setpoint) / s
    if z > 60.0:
        return 1.0
    if z < -60.0:
        return 0.0
    return 1.0 / (1.0 + float(np.exp(-z)))


def hamilton_transfer_terms(
    donor_energy: float,
    recip_energy: float,
    r_dr: float,
    transfer: float,
    recip_setpoint: float,
    recip_scale: float,
    donor_setpoint: float,
    donor_scale: float,
) -> Tuple[float, float, float, float, float]:
    b_r = logistic_proxy(recip_energy + transfer, recip_setpoint, recip_scale) - logistic_proxy(
        recip_energy, recip_setpoint, recip_scale
    )
    c_d = logistic_proxy(donor_energy, donor_setpoint, donor_scale) - logistic_proxy(
        max(0.0, donor_energy - transfer), donor_setpoint, donor_scale
    )
    lhs = r_dr * b_r
    margin = lhs - c_d
    return r_dr, b_r, c_d, lhs, margin


def estimate_relatedness_from_cues(
    donor: Predator | Prey,
    recip: Predator | Prey,
    cue_memory: Dict[int, Dict[int, float]],
) -> Tuple[float, float, float]:
    if KIN_CUE_MODEL != "maternal_residence":
        raise ValueError(f"Unknown KIN_CUE_MODEL: {KIN_CUE_MODEL}")

    shared_mother = (
        donor.mother_id is not None
        and recip.mother_id is not None
        and donor.mother_id == recip.mother_id
    )
    if shared_mother:
        maternal_signal = 1.0 if random.random() < MATERNAL_CUE_TPR else 0.0
    else:
        maternal_signal = 1.0 if random.random() < MATERNAL_CUE_FPR else 0.0

    co_res_score = cue_memory.get(donor.aid, {}).get(recip.aid, 0.0)
    coresidence_signal = min(1.0, co_res_score / max(1e-9, CORESIDENCE_SATURATION))

    r_hat = (
        KIN_CONSERVATIVE_BIAS
        + KIN_W_MATERNAL * maternal_signal
        + KIN_W_CORESIDENCE * coresidence_signal
    )
    r_hat = float(np.clip(r_hat, 0.0, 1.0))
    return r_hat, maternal_signal, coresidence_signal


def kin_transfer_phase(
    agents: List[Predator | Prey],
    rel: Dict[int, Dict[int, float]],
    cue_memory: Dict[int, Dict[int, float]],
    dependent_age_max: int,
    transfer_radius: int,
    transfer_chunk: float,
    max_chunks_per_donor: int,
    reserve_energy: float,
    recip_surv_setpoint: float,
    recip_surv_scale: float,
    donor_fit_setpoint: float,
    donor_fit_scale: float,
    stats: Dict[str, float],
    prefix: str,
) -> None:
    alive = [ag for ag in agents if ag.energy > 0.0]
    if not alive:
        return

    index = build_spatial_index(alive)
    random.shuffle(alive)

    for donor in alive:
        chunks_done = 0
        while chunks_done < max_chunks_per_donor:
            available = donor.energy - reserve_energy
            if available <= 0.0:
                break
            transfer = min(transfer_chunk, available)
            if transfer <= 0.0:
                break

            cand_idx = nearby_indices(index, donor.x, donor.y, transfer_radius)
            dep_candidates: List[Tuple[Predator | Prey, float, float, float, float]] = []
            other_candidates: List[Tuple[Predator | Prey, float, float, float, float]] = []

            for idx in cand_idx:
                recip = alive[idx]
                if recip.aid == donor.aid or recip.energy <= 0.0:
                    continue

                r_true = coefficient_of_relationship(donor.aid, recip.aid, rel)
                r_hat, maternal_cue, coresidence_cue = estimate_relatedness_from_cues(donor, recip, cue_memory)

                if recip.age <= dependent_age_max:
                    dep_candidates.append((recip, r_hat, r_true, maternal_cue, coresidence_cue))
                else:
                    other_candidates.append((recip, r_hat, r_true, maternal_cue, coresidence_cue))

            if DEPENDENT_PRIORITY and dep_candidates:
                candidates = dep_candidates
            else:
                candidates = dep_candidates + other_candidates

            if not candidates:
                break

            best: Tuple[Predator | Prey, float, float, float, float, float, float] | None = None
            for recip, r_hat, r_true, maternal_cue, coresidence_cue in candidates:
                r_val, b_val, c_val, lhs, margin = hamilton_transfer_terms(
                    donor.energy,
                    recip.energy,
                    r_hat,
                    transfer,
                    recip_surv_setpoint,
                    recip_surv_scale,
                    donor_fit_setpoint,
                    donor_fit_scale,
                )
                if best is None or margin > best[5]:
                    best = (recip, r_val, b_val, c_val, lhs, margin, r_true)

            if best is None:
                break

            recip, r_val, b_val, c_val, lhs, margin, r_true = best
            stats[f"{prefix}_decisions"] += 1.0
            stats[f"{prefix}_r_sum"] += r_val
            stats[f"{prefix}_r_hat_sum"] += r_val
            stats[f"{prefix}_r_true_sum"] += r_true
            stats[f"{prefix}_r_abs_err_sum"] += abs(r_val - r_true)
            stats[f"{prefix}_b_sum"] += b_val
            stats[f"{prefix}_c_sum"] += c_val
            stats[f"{prefix}_lhs_sum"] += lhs
            stats[f"{prefix}_margin_sum"] += margin

            est_pos = r_val >= EST_KIN_THRESHOLD
            true_pos = r_true >= TRUE_KIN_THRESHOLD
            if est_pos and true_pos:
                stats[f"{prefix}_tp_count"] += 1.0
            elif est_pos and not true_pos:
                stats[f"{prefix}_fp_count"] += 1.0
            elif (not est_pos) and true_pos:
                stats[f"{prefix}_fn_count"] += 1.0
            else:
                stats[f"{prefix}_tn_count"] += 1.0

            if margin <= 0.0:
                break
            if donor.energy - transfer < reserve_energy:
                break

            donor.energy -= transfer
            recip.energy += transfer
            chunks_done += 1

            stats[f"{prefix}_accepts"] += 1.0
            stats[f"{prefix}_energy"] += transfer
            if r_true >= TRUE_KIN_THRESHOLD:
                stats[f"{prefix}_help_kin_energy"] += transfer
            else:
                stats[f"{prefix}_help_nonkin_energy"] += transfer


# ============================================================
# ECOLOGY PHASES
# ============================================================

def prey_phase(preys: List[Prey], grass: np.ndarray) -> List[Prey]:
    random.shuffle(preys)
    for pr in preys:
        pr.age += 1

        moved = False
        if random.random() < PREY_MOVE_PROB:
            pr.x = wrap(pr.x + random.choice([-1, 0, 1]), W)
            pr.y = wrap(pr.y + random.choice([-1, 0, 1]), H)
            moved = True

        pr.energy, _ = drain_energy(pr.energy, PREY_METAB)
        if moved:
            pr.energy, _ = drain_energy(pr.energy, PREY_MOVE_COST)

        if pr.energy <= 0.0:
            continue

        bite = min(PREY_BITE_SIZE, float(grass[pr.y, pr.x]))
        if bite > 0.0:
            grass[pr.y, pr.x] -= bite
            pr.energy += bite

    return preys


def predator_phase(preds: List[Predator]) -> List[Predator]:
    random.shuffle(preds)
    for pd in preds:
        pd.age += 1

        moved = False
        if random.random() < PRED_MOVE_PROB:
            pd.x = wrap(pd.x + random.choice([-1, 0, 1]), W)
            pd.y = wrap(pd.y + random.choice([-1, 0, 1]), H)
            moved = True

        pd.energy, _ = drain_energy(pd.energy, PRED_METAB)
        if moved:
            pd.energy, _ = drain_energy(pd.energy, PRED_MOVE_COST)

    return preds


def resolve_solo_predation(preds: List[Predator], preys: List[Prey], eco_stats: Dict[str, float]) -> set[int]:
    prey_index = build_spatial_index(preys)
    prey_killed: set[int] = set()

    pred_order = list(range(len(preds)))
    random.shuffle(pred_order)

    for pidx in pred_order:
        pred = preds[pidx]
        if pred.energy <= 0.0:
            continue

        candidate_prey = nearby_indices(prey_index, pred.x, pred.y, PRED_HUNT_R)
        candidate_prey = [i for i in candidate_prey if i not in prey_killed and preys[i].energy > 0.0]
        if not candidate_prey:
            continue

        victim_idx = random.choice(candidate_prey)
        victim = preys[victim_idx]
        prey_energy = max(0.0, victim.energy)

        pkill = 1.0 - float(np.exp(-PRED_SOLO_HUNT_SCALE * pred.energy / max(prey_energy, 1e-9)))
        pkill = max(0.0, min(1.0, pkill))
        if random.random() >= pkill:
            continue

        prey_killed.add(victim_idx)
        gain = prey_energy * PRED_HUNT_YIELD
        pred.energy += gain

        eco_stats["solo_kills"] += 1.0
        eco_stats["prey_to_pred_energy"] += gain

    return prey_killed


def alive_predators(preds: List[Predator]) -> List[Predator]:
    return [pd for pd in preds if pd.energy > 0.0 and pd.age <= PRED_LIFE_EXPECTANCY]


def alive_preys(preys: List[Prey]) -> List[Prey]:
    return [pr for pr in preys if pr.energy > 0.0 and pr.age <= PREY_LIFE_EXPECTANCY]


# ============================================================
# SEXUAL REPRODUCTION
# ============================================================

def pick_best_male(
    female_x: int,
    female_y: int,
    agents: List[Predator | Prey],
    index: Dict[Tuple[int, int], List[int]],
    mate_radius: int,
    male_mature_age: int,
    male_min_energy: float,
) -> Predator | Prey | None:
    candidate_idx = nearby_indices(index, female_x, female_y, mate_radius)
    males = [
        agents[i]
        for i in candidate_idx
        if agents[i].sex == "M"
        and agents[i].age >= male_mature_age
        and agents[i].energy >= male_min_energy
    ]
    if not males:
        return None
    # Female choice: prefer the highest-energy male.
    return max(males, key=lambda m: m.energy)


def reproduce_predators(
    preds: List[Predator],
    rel: Dict[int, Dict[int, float]],
    next_pred_id: int,
) -> Tuple[List[Predator], int]:
    if not preds:
        return preds, next_pred_id

    crowd_scale = max(0.0, 1.0 - len(preds) / max(1, PRED_MAX))
    index = build_spatial_index(preds)
    existing_ids = [pd.aid for pd in preds]
    newborn: List[Predator] = []

    female_order = [
        pd for pd in preds
        if pd.sex == "F"
        and PRED_FEMALE_FERTILE_MIN <= pd.age <= PRED_FEMALE_FERTILE_MAX
        and pd.energy >= PRED_BIRTH_THRESH
    ]
    random.shuffle(female_order)

    for female in female_order:
        if random.random() >= PRED_REPRO_PROB * crowd_scale:
            continue

        male = pick_best_male(
            female.x,
            female.y,
            preds,
            index,
            PRED_MATE_R,
            PRED_MALE_MATURE_AGE,
            PRED_MALE_MIN_ENERGY,
        )
        if male is None:
            continue

        child_energy = female.energy * PRED_CHILD_SHARE
        if child_energy <= 0.0:
            continue

        female.energy -= child_energy
        male.energy, _ = drain_energy(male.energy, PRED_FATHER_BIRTH_COST)

        child = Predator(
            aid=next_pred_id,
            x=wrap(female.x + random.randint(-1, 1), W),
            y=wrap(female.y + random.randint(-1, 1), H),
            energy=child_energy,
            sex=random_sex(),
            age=0,
            mother_id=female.aid,
            father_id=male.aid,
        )
        next_pred_id += 1
        newborn.append(child)

        add_child_relatedness(rel, child.aid, female.aid, male.aid, existing_ids)
        existing_ids.append(child.aid)

    out = preds + newborn
    if len(out) > PRED_MAX:
        random.shuffle(out)
        out = out[:PRED_MAX]
    return out, next_pred_id


def reproduce_preys(
    preys: List[Prey],
    rel: Dict[int, Dict[int, float]],
    next_prey_id: int,
) -> Tuple[List[Prey], int]:
    if not preys:
        return preys, next_prey_id

    crowd_scale = max(0.0, 1.0 - len(preys) / max(1, PREY_MAX))
    index = build_spatial_index(preys)
    existing_ids = [pr.aid for pr in preys]
    newborn: List[Prey] = []

    female_order = [
        pr for pr in preys
        if pr.sex == "F"
        and PREY_FEMALE_FERTILE_MIN <= pr.age <= PREY_FEMALE_FERTILE_MAX
        and pr.energy >= PREY_BIRTH_THRESH
    ]
    random.shuffle(female_order)

    for female in female_order:
        if random.random() >= PREY_REPRO_PROB * crowd_scale:
            continue

        male = pick_best_male(
            female.x,
            female.y,
            preys,
            index,
            PREY_MATE_R,
            PREY_MALE_MATURE_AGE,
            PREY_MALE_MIN_ENERGY,
        )
        if male is None:
            continue

        child_energy = female.energy * PREY_CHILD_SHARE
        if child_energy <= 0.0:
            continue

        female.energy -= child_energy
        male.energy, _ = drain_energy(male.energy, PREY_FATHER_BIRTH_COST)

        child = Prey(
            aid=next_prey_id,
            x=wrap(female.x + random.randint(-1, 1), W),
            y=wrap(female.y + random.randint(-1, 1), H),
            energy=child_energy,
            sex=random_sex(),
            age=0,
            mother_id=female.aid,
            father_id=male.aid,
        )
        next_prey_id += 1
        newborn.append(child)

        add_child_relatedness(rel, child.aid, female.aid, male.aid, existing_ids)
        existing_ids.append(child.aid)

    out = preys + newborn
    if len(out) > PREY_MAX:
        random.shuffle(out)
        out = out[:PREY_MAX]
    return out, next_prey_id


# ============================================================
# WORLD STEP
# ============================================================

def step_world(
    preds: List[Predator],
    preys: List[Prey],
    grass: np.ndarray,
    pred_rel: Dict[int, Dict[int, float]],
    prey_rel: Dict[int, Dict[int, float]],
    pred_cue_memory: Dict[int, Dict[int, float]],
    prey_cue_memory: Dict[int, Dict[int, float]],
    next_pred_id: int,
    next_prey_id: int,
    step_stats: Dict[str, float],
) -> Tuple[List[Predator], List[Prey], np.ndarray, int, int]:
    # Grass regrowth.
    np.minimum(grass + GRASS_REGROWTH, GRASS_MAX, out=grass)

    # Core ecology.
    preys = prey_phase(preys, grass)
    preds = predator_phase(preds)

    prey_killed = resolve_solo_predation(preds, preys, step_stats)
    if prey_killed:
        preys = [pr for i, pr in enumerate(preys) if i not in prey_killed]

    # Age/starvation cleanup before reproduction.
    preds = alive_predators(preds)
    preys = alive_preys(preys)
    prune_relatedness(pred_rel, {pd.aid for pd in preds})
    prune_relatedness(prey_rel, {pr.aid for pr in preys})

    # Sexual reproduction.
    preys, next_prey_id = reproduce_preys(preys, prey_rel, next_prey_id)
    preds, next_pred_id = reproduce_predators(preds, pred_rel, next_pred_id)

    # Cue-memory maintenance before kin-transfer decisions.
    pred_alive_ids_pre_transfer = {pd.aid for pd in preds if pd.energy > 0.0}
    prey_alive_ids_pre_transfer = {pr.aid for pr in preys if pr.energy > 0.0}
    decay_cue_memory(pred_cue_memory, CORESIDENCE_DECAY, pred_alive_ids_pre_transfer)
    decay_cue_memory(prey_cue_memory, CORESIDENCE_DECAY, prey_alive_ids_pre_transfer)
    update_co_residence_memory(preds, pred_cue_memory, CORESIDENCE_RADIUS_PRED, CORESIDENCE_INCREMENT)
    update_co_residence_memory(preys, prey_cue_memory, CORESIDENCE_RADIUS_PREY, CORESIDENCE_INCREMENT)

    # Kin-selection cooperation: energy transfer only.
    kin_transfer_phase(
        preds,
        pred_rel,
        pred_cue_memory,
        dependent_age_max=PRED_DEPENDENT_AGE_MAX,
        transfer_radius=PRED_KIN_TRANSFER_R,
        transfer_chunk=PRED_KIN_TRANSFER_CHUNK,
        max_chunks_per_donor=PRED_KIN_MAX_CHUNKS_PER_DONOR,
        reserve_energy=PRED_KIN_RESERVE,
        recip_surv_setpoint=PRED_RECIP_SURV_SETPOINT,
        recip_surv_scale=PRED_RECIP_SURV_SCALE,
        donor_fit_setpoint=PRED_DONOR_FIT_SETPOINT,
        donor_fit_scale=PRED_DONOR_FIT_SCALE,
        stats=step_stats,
        prefix="pred_kin",
    )

    kin_transfer_phase(
        preys,
        prey_rel,
        prey_cue_memory,
        dependent_age_max=PREY_DEPENDENT_AGE_MAX,
        transfer_radius=PREY_KIN_TRANSFER_R,
        transfer_chunk=PREY_KIN_TRANSFER_CHUNK,
        max_chunks_per_donor=PREY_KIN_MAX_CHUNKS_PER_DONOR,
        reserve_energy=PREY_KIN_RESERVE,
        recip_surv_setpoint=PREY_RECIP_SURV_SETPOINT,
        recip_surv_scale=PREY_RECIP_SURV_SCALE,
        donor_fit_setpoint=PREY_DONOR_FIT_SETPOINT,
        donor_fit_scale=PREY_DONOR_FIT_SCALE,
        stats=step_stats,
        prefix="prey_kin",
    )

    # Final cleanup.
    preds = alive_predators(preds)
    preys = alive_preys(preys)
    prune_relatedness(pred_rel, {pd.aid for pd in preds})
    prune_relatedness(prey_rel, {pr.aid for pr in preys})
    prune_cue_memory(pred_cue_memory, {pd.aid for pd in preds})
    prune_cue_memory(prey_cue_memory, {pr.aid for pr in preys})

    return preds, preys, grass, next_pred_id, next_prey_id


# ============================================================
# RUN / REPORT
# ============================================================

def run_sim(seed_override: int | None = None) -> Tuple[
    List[int],
    List[int],
    Dict[str, List[float]],
    List[Predator],
    List[Prey],
    bool,
    int | None,
]:
    if seed_override is not None:
        random.seed(seed_override)
        np.random.seed(seed_override)
    elif SEED is not None:
        random.seed(SEED)
        np.random.seed(SEED)

    preds: List[Predator] = [
        Predator(
            aid=i,
            x=random.randrange(W),
            y=random.randrange(H),
            energy=PRED_ENERGY_INIT,
            sex=random_sex(),
            age=random.randint(PRED_FEMALE_FERTILE_MIN, PRED_FEMALE_FERTILE_MAX),
        )
        for i in range(PRED_INIT)
    ]
    next_pred_id = PRED_INIT

    preys: List[Prey] = [
        Prey(
            aid=i,
            x=random.randrange(W),
            y=random.randrange(H),
            energy=sample_prey_energy(),
            sex=random_sex(),
            age=random.randint(PREY_FEMALE_FERTILE_MIN, PREY_FEMALE_FERTILE_MAX),
        )
        for i in range(PREY_INIT)
    ]
    next_prey_id = PREY_INIT

    pred_rel = init_relatedness_matrix([pd.aid for pd in preds])
    prey_rel = init_relatedness_matrix([pr.aid for pr in preys])
    pred_cue_memory = init_cue_memory([pd.aid for pd in preds])
    prey_cue_memory = init_cue_memory([pr.aid for pr in preys])
    seed_founder_clans(pred_rel, [pd.aid for pd in preds], PRED_FOUNDER_CLAN_SIZE, FOUNDER_CLAN_R)
    seed_founder_clans(prey_rel, [pr.aid for pr in preys], PREY_FOUNDER_CLAN_SIZE, FOUNDER_CLAN_R)

    grass = init_grass()

    pred_hist: List[int] = []
    prey_hist: List[int] = []

    kin_hist: Dict[str, List[float]] = {
        "pred_transfer_rate": [],
        "pred_mean_r": [],
        "pred_mean_b": [],
        "pred_mean_c": [],
        "pred_mean_lhs": [],
        "pred_mean_margin": [],
        "pred_energy_transferred": [],
        "pred_true_r_mean": [],
        "pred_r_abs_err_mean": [],
        "pred_false_pos_rate": [],
        "pred_false_neg_rate": [],
        "pred_help_nonkin_energy": [],
        "pred_help_kin_energy": [],
        "prey_transfer_rate": [],
        "prey_mean_r": [],
        "prey_mean_b": [],
        "prey_mean_c": [],
        "prey_mean_lhs": [],
        "prey_mean_margin": [],
        "prey_energy_transferred": [],
        "prey_true_r_mean": [],
        "prey_r_abs_err_mean": [],
        "prey_false_pos_rate": [],
        "prey_false_neg_rate": [],
        "prey_help_nonkin_energy": [],
        "prey_help_kin_energy": [],
        "solo_kills": [],
        "prey_to_pred_energy": [],
    }

    extinction_step: int | None = None

    for t in range(STEPS):
        step_stats = {
            "pred_kin_decisions": 0.0,
            "pred_kin_accepts": 0.0,
            "pred_kin_r_sum": 0.0,
            "pred_kin_r_hat_sum": 0.0,
            "pred_kin_r_true_sum": 0.0,
            "pred_kin_r_abs_err_sum": 0.0,
            "pred_kin_b_sum": 0.0,
            "pred_kin_c_sum": 0.0,
            "pred_kin_lhs_sum": 0.0,
            "pred_kin_margin_sum": 0.0,
            "pred_kin_energy": 0.0,
            "pred_kin_tp_count": 0.0,
            "pred_kin_fp_count": 0.0,
            "pred_kin_fn_count": 0.0,
            "pred_kin_tn_count": 0.0,
            "pred_kin_help_nonkin_energy": 0.0,
            "pred_kin_help_kin_energy": 0.0,
            "prey_kin_decisions": 0.0,
            "prey_kin_accepts": 0.0,
            "prey_kin_r_sum": 0.0,
            "prey_kin_r_hat_sum": 0.0,
            "prey_kin_r_true_sum": 0.0,
            "prey_kin_r_abs_err_sum": 0.0,
            "prey_kin_b_sum": 0.0,
            "prey_kin_c_sum": 0.0,
            "prey_kin_lhs_sum": 0.0,
            "prey_kin_margin_sum": 0.0,
            "prey_kin_energy": 0.0,
            "prey_kin_tp_count": 0.0,
            "prey_kin_fp_count": 0.0,
            "prey_kin_fn_count": 0.0,
            "prey_kin_tn_count": 0.0,
            "prey_kin_help_nonkin_energy": 0.0,
            "prey_kin_help_kin_energy": 0.0,
            "solo_kills": 0.0,
            "prey_to_pred_energy": 0.0,
        }

        preds, preys, grass, next_pred_id, next_prey_id = step_world(
            preds,
            preys,
            grass,
            pred_rel,
            prey_rel,
            pred_cue_memory,
            prey_cue_memory,
            next_pred_id,
            next_prey_id,
            step_stats,
        )

        pred_n = len(preds)
        prey_n = len(preys)
        pred_hist.append(pred_n)
        prey_hist.append(prey_n)

        pred_decisions = step_stats["pred_kin_decisions"]
        if pred_decisions > 0:
            pred_transfer_rate = step_stats["pred_kin_accepts"] / pred_decisions
            pred_mean_r = step_stats["pred_kin_r_hat_sum"] / pred_decisions
            pred_true_r_mean = step_stats["pred_kin_r_true_sum"] / pred_decisions
            pred_r_abs_err_mean = step_stats["pred_kin_r_abs_err_sum"] / pred_decisions
            pred_mean_b = step_stats["pred_kin_b_sum"] / pred_decisions
            pred_mean_c = step_stats["pred_kin_c_sum"] / pred_decisions
            pred_mean_lhs = step_stats["pred_kin_lhs_sum"] / pred_decisions
            pred_mean_margin = step_stats["pred_kin_margin_sum"] / pred_decisions
        else:
            pred_transfer_rate = pred_mean_r = pred_mean_b = pred_mean_c = 0.0
            pred_true_r_mean = pred_r_abs_err_mean = 0.0
            pred_mean_lhs = pred_mean_margin = 0.0
        pred_fp_denom = step_stats["pred_kin_fp_count"] + step_stats["pred_kin_tn_count"]
        pred_fn_denom = step_stats["pred_kin_fn_count"] + step_stats["pred_kin_tp_count"]
        pred_false_pos_rate = (step_stats["pred_kin_fp_count"] / pred_fp_denom) if pred_fp_denom > 0.0 else 0.0
        pred_false_neg_rate = (step_stats["pred_kin_fn_count"] / pred_fn_denom) if pred_fn_denom > 0.0 else 0.0

        prey_decisions = step_stats["prey_kin_decisions"]
        if prey_decisions > 0:
            prey_transfer_rate = step_stats["prey_kin_accepts"] / prey_decisions
            prey_mean_r = step_stats["prey_kin_r_hat_sum"] / prey_decisions
            prey_true_r_mean = step_stats["prey_kin_r_true_sum"] / prey_decisions
            prey_r_abs_err_mean = step_stats["prey_kin_r_abs_err_sum"] / prey_decisions
            prey_mean_b = step_stats["prey_kin_b_sum"] / prey_decisions
            prey_mean_c = step_stats["prey_kin_c_sum"] / prey_decisions
            prey_mean_lhs = step_stats["prey_kin_lhs_sum"] / prey_decisions
            prey_mean_margin = step_stats["prey_kin_margin_sum"] / prey_decisions
        else:
            prey_transfer_rate = prey_mean_r = prey_mean_b = prey_mean_c = 0.0
            prey_true_r_mean = prey_r_abs_err_mean = 0.0
            prey_mean_lhs = prey_mean_margin = 0.0
        prey_fp_denom = step_stats["prey_kin_fp_count"] + step_stats["prey_kin_tn_count"]
        prey_fn_denom = step_stats["prey_kin_fn_count"] + step_stats["prey_kin_tp_count"]
        prey_false_pos_rate = (step_stats["prey_kin_fp_count"] / prey_fp_denom) if prey_fp_denom > 0.0 else 0.0
        prey_false_neg_rate = (step_stats["prey_kin_fn_count"] / prey_fn_denom) if prey_fn_denom > 0.0 else 0.0

        kin_hist["pred_transfer_rate"].append(pred_transfer_rate)
        kin_hist["pred_mean_r"].append(pred_mean_r)
        kin_hist["pred_true_r_mean"].append(pred_true_r_mean)
        kin_hist["pred_r_abs_err_mean"].append(pred_r_abs_err_mean)
        kin_hist["pred_false_pos_rate"].append(pred_false_pos_rate)
        kin_hist["pred_false_neg_rate"].append(pred_false_neg_rate)
        kin_hist["pred_mean_b"].append(pred_mean_b)
        kin_hist["pred_mean_c"].append(pred_mean_c)
        kin_hist["pred_mean_lhs"].append(pred_mean_lhs)
        kin_hist["pred_mean_margin"].append(pred_mean_margin)
        kin_hist["pred_energy_transferred"].append(step_stats["pred_kin_energy"])
        kin_hist["pred_help_nonkin_energy"].append(step_stats["pred_kin_help_nonkin_energy"])
        kin_hist["pred_help_kin_energy"].append(step_stats["pred_kin_help_kin_energy"])

        kin_hist["prey_transfer_rate"].append(prey_transfer_rate)
        kin_hist["prey_mean_r"].append(prey_mean_r)
        kin_hist["prey_true_r_mean"].append(prey_true_r_mean)
        kin_hist["prey_r_abs_err_mean"].append(prey_r_abs_err_mean)
        kin_hist["prey_false_pos_rate"].append(prey_false_pos_rate)
        kin_hist["prey_false_neg_rate"].append(prey_false_neg_rate)
        kin_hist["prey_mean_b"].append(prey_mean_b)
        kin_hist["prey_mean_c"].append(prey_mean_c)
        kin_hist["prey_mean_lhs"].append(prey_mean_lhs)
        kin_hist["prey_mean_margin"].append(prey_mean_margin)
        kin_hist["prey_energy_transferred"].append(step_stats["prey_kin_energy"])
        kin_hist["prey_help_nonkin_energy"].append(step_stats["prey_kin_help_nonkin_energy"])
        kin_hist["prey_help_kin_energy"].append(step_stats["prey_kin_help_kin_energy"])

        kin_hist["solo_kills"].append(step_stats["solo_kills"])
        kin_hist["prey_to_pred_energy"].append(step_stats["prey_to_pred_energy"])

        if (t + 1) % REPORT_EVERY == 0:
            print(
                f"t={t+1:4d} preds={pred_n:4d} preys={prey_n:4d} "
                f"solo_kills={step_stats['solo_kills']:.0f} "
                f"pred_transfer_rate={pred_transfer_rate:.3f} pred_transfer_E={step_stats['pred_kin_energy']:.3f} "
                f"prey_transfer_rate={prey_transfer_rate:.3f} prey_transfer_E={step_stats['prey_kin_energy']:.3f}"
            )

        if pred_n == 0 or prey_n == 0:
            extinction_step = t + 1
            print(f"Extinction at step {extinction_step}: preds={pred_n} preys={prey_n}")
            break

    success = extinction_step is None
    return pred_hist, prey_hist, kin_hist, preds, preys, success, extinction_step


# ============================================================
# PLOTS
# ============================================================

def plot_populations(pred_hist: List[int], prey_hist: List[int]) -> None:
    plt.figure()
    plt.plot(prey_hist, label="Prey")
    plt.plot(pred_hist, label="Predators")
    plt.xlabel("Time step")
    plt.ylabel("Count")
    plt.title("Predator-prey dynamics (solo hunt + kin-transfer cooperation)")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.show()


def plot_kin_diagnostics(kin_hist: Dict[str, List[float]]) -> None:
    t = np.arange(1, len(kin_hist.get("pred_transfer_rate", [])) + 1)
    if len(t) == 0:
        print("No kin diagnostics to plot.")
        return

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10.0, 8.5), sharex=True)

    ax1.plot(t, kin_hist["pred_transfer_rate"], label="Predator kin transfer rate")
    ax1.plot(t, kin_hist["prey_transfer_rate"], label="Prey kin transfer rate")
    ax1.set_ylabel("Transfer rate")
    ax1.set_ylim(0.0, 1.0)
    ax1.set_title("Hamilton Kin-Selection Diagnostics")
    ax1.grid(alpha=0.25)
    ax1.legend(loc="upper right")

    ax2.plot(t, kin_hist["pred_mean_margin"], label="Predator mean(rB-C)")
    ax2.plot(t, kin_hist["prey_mean_margin"], label="Prey mean(rB-C)")
    ax2.set_ylabel("Margin")
    ax2.grid(alpha=0.25)
    ax2.legend(loc="upper right")

    ax3.plot(t, kin_hist["pred_energy_transferred"], label="Predator kin energy transferred")
    ax3.plot(t, kin_hist["prey_energy_transferred"], label="Prey kin energy transferred")
    ax3.plot(t, kin_hist["solo_kills"], label="Solo predator kills")
    ax3.set_xlabel("Time step")
    ax3.set_ylabel("Energy / count")
    ax3.grid(alpha=0.25)
    ax3.legend(loc="upper right")

    fig.tight_layout()
    plt.show()


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    attempts = 0
    while True:
        seed = None if SEED is None else SEED + attempts
        (
            pred_hist,
            prey_hist,
            kin_hist,
            preds_final,
            preys_final,
            success,
            extinction_step,
        ) = run_sim(seed_override=seed)

        if not RESTART_ON_EXTINCTION or success:
            break

        attempts += 1
        if attempts > MAX_RESTARTS:
            print(
                f"Failed to reach {STEPS} steps after {MAX_RESTARTS} restarts "
                f"(last extinction at step {extinction_step})."
            )
            break
        print(f"Restarting (attempt {attempts}/{MAX_RESTARTS})...")

    print(f"Final populations: predators={len(preds_final)} prey={len(preys_final)}")

    if PLOT_RESULTS:
        plot_populations(pred_hist, prey_hist)
        plot_kin_diagnostics(kin_hist)


if __name__ == "__main__":
    main()
