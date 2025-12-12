# planning_state.py
from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np


@dataclass
class EnvSnapshot:
    timestep: int
    agents: List[str]
    agents_positions: List[Tuple[int, int]]
    persons_set: Any
    # ProbabilityMatrix state (explicit)
    pm_supposed_position: Tuple[int, int]
    pm_map: np.ndarray
    pm_map_prob: np.ndarray
    pm_inc_x: float
    pm_inc_y: float
    pm_spacement: float
    pm_time_step_relation: int
    pm_movement_vector: Tuple[float, float]
    pm_amplitude: int
    pm_spacement_inc: float
    # optional reproducibility for env.step() randomness
    py_random_state: object


def save_env_state(env) -> EnvSnapshot:
    """
    Save minimal env state required to rollback between planner simulations.
    Requires env.probability_matrix to exist (i.e., env.reset() has been called).
    """
    if not hasattr(env, "probability_matrix") or env.probability_matrix is None:
        raise ValueError("env.probability_matrix is None — did you call env.reset()?")

    pm = env.probability_matrix

    return EnvSnapshot(
        timestep=int(env.timestep),
        agents=list(env.agents),
        agents_positions=list(env.agents_positions),
        persons_set=copy.deepcopy(env.persons_set),
        pm_supposed_position=tuple(pm.supposed_position),
        pm_map=pm.map.copy(),
        pm_map_prob=pm.map_prob.copy(),
        pm_inc_x=float(pm.inc_x),
        pm_inc_y=float(pm.inc_y),
        pm_spacement=float(pm.spacement),
        pm_time_step_relation=int(pm.time_step_relation),
        pm_movement_vector=tuple(pm.movement_vector),
        pm_amplitude=int(pm.amplitude),
        pm_spacement_inc=float(pm.spacement_inc),
        py_random_state=random.getstate(),
    )


def load_env_state(env, snap: EnvSnapshot) -> None:
    """
    Restore env state from snapshot.
    """
    env.timestep = int(snap.timestep)
    env.agents = list(snap.agents)
    env.agents_positions = list(snap.agents_positions)
    env.persons_set = copy.deepcopy(snap.persons_set)

    pm = env.probability_matrix

    # Restore ProbabilityMatrix fields
    pm.supposed_position = tuple(snap.pm_supposed_position)
    pm.map = snap.pm_map.copy()
    pm.map_prob = snap.pm_map_prob.copy()
    pm.inc_x = float(snap.pm_inc_x)
    pm.inc_y = float(snap.pm_inc_y)
    pm.spacement = float(snap.pm_spacement)
    pm.time_step_relation = int(snap.pm_time_step_relation)
    pm.movement_vector = tuple(snap.pm_movement_vector)
    pm.amplitude = int(snap.pm_amplitude)
    pm.spacement_inc = float(snap.pm_spacement_inc)

    random.setstate(snap.py_random_state)
