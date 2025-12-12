# sparse_sampling.py
from __future__ import annotations

import itertools
from typing import Dict, List, Tuple, Sequence

from DSSE.environment.constants import Actions
from DSSE.environment.planning_state import save_env_state, load_env_state


def _enumerate_joint_actions(env, agents: Sequence[str]) -> List[Tuple[int, ...]]:
    if not agents:
        return []
    nA = env.action_space(agents[0]).n
    return list(itertools.product(range(nA), repeat=len(agents)))


def _predict_next_positions(env, agents: Sequence[str], joint_action: Tuple[int, ...]):
    """
    1-step lookahead collision filter (deterministic): prevent same-cell next step.
    """
    pos_now = list(env.agents_positions)
    idx_map = {a: i for i, a in enumerate(env.agents)}
    next_positions = []
    for a_id, a in zip(agents, joint_action):
        i = idx_map[a_id]
        x, y = pos_now[i]
        if int(a) == Actions.SEARCH.value:
            next_positions.append((x, y))
        else:
            next_positions.append(env.move_drone((x, y), int(a)))
    return next_positions


def _safe_joint_actions(env, agents: Sequence[str]) -> List[Tuple[int, ...]]:
    all_ja = _enumerate_joint_actions(env, agents)
    safe = []
    for ja in all_ja:
        nxt = _predict_next_positions(env, agents, ja)
        if len(set(nxt)) == len(nxt):
            safe.append(ja)
    return safe or all_ja


def _mean_reward(rewards: Dict, agents: Sequence[str]) -> float:
    if not rewards:
        return 0.0
    return sum(float(rewards[a]) for a in rewards.keys()) / max(1, len(agents))


def _sparse_value(
    env,
    agents: Sequence[str],
    depth: int,
    samples: int,
    debug: bool,
    indent: int,
) -> float:
    if depth == 0:
        return 0.0

    joint_actions = _safe_joint_actions(env, agents)
    best_val = float("-inf")

    if debug and indent <= 2:
        print("  " * indent + f"[SS] depth={depth}, joint_actions={len(joint_actions)}")

    for ja in joint_actions:
        total = 0.0

        for s in range(samples):
            snap = save_env_state(env)

            acts = {a_id: int(a) for a_id, a in zip(agents, ja)}
            obs, rewards, terms, truncs, infos = env.step(acts)

            r = _mean_reward(rewards, agents)
            done = any(terms.values()) or any(truncs.values())

            if done:
                total += r
            else:
                total += r + _sparse_value(
                    env, agents, depth - 1, samples, debug, indent + 1
                )

            load_env_state(env, snap)

        avg = total / samples
        if avg > best_val:
            best_val = avg

    return best_val


def sparse_sampling_plan(
    env,
    agents: Sequence[str],
    depth: int = 2,
    samples: int = 3,
    debug: bool = True,
) -> Dict[str, int]:
    """
    Sparse Sampling planner (centralized joint action) without deepcopy().
    """
    if not agents:
        return {}

    joint_actions = _safe_joint_actions(env, agents)

    if debug:
        print(
            f"\n=== SparseSampling plan: depth={depth}, samples={samples}, joint_actions={len(joint_actions)} ==="
        )

    best_ja = None
    best_val = float("-inf")

    for i, ja in enumerate(joint_actions):
        total = 0.0

        # light top-level progress
        if debug:
            print(f"[SS] eval {i+1}/{len(joint_actions)} JA={ja}")

        for s in range(samples):
            snap = save_env_state(env)

            acts = {a_id: int(a) for a_id, a in zip(agents, ja)}
            obs, rewards, terms, truncs, infos = env.step(acts)

            r = _mean_reward(rewards, agents)
            done = any(terms.values()) or any(truncs.values())

            if done:
                total += r
            else:
                total += r + _sparse_value(
                    env, agents, depth - 1, samples, debug, indent=1
                )

            load_env_state(env, snap)

        avg = total / samples
        if debug:
            print(f"[SS]   JA={ja} -> Q~{avg:.3f}")

        if avg > best_val or best_ja is None:
            best_val = avg
            best_ja = ja
            if debug:
                print(f"[SS]   NEW BEST: JA={best_ja}, Q~{best_val:.3f}")

    return {a_id: int(a) for a_id, a in zip(agents, best_ja)}
