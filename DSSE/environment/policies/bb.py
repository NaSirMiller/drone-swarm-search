# branch_and_bound.py
from __future__ import annotations

import itertools
from typing import Dict, Sequence, Tuple, List

from DSSE.environment.constants import Actions
from DSSE.environment.planning_state import save_env_state, load_env_state


def _enumerate_joint_actions(env, agents: Sequence[str]) -> List[Tuple[int, ...]]:
    nA = env.action_space(agents[0]).n
    return list(itertools.product(range(nA), repeat=len(agents)))


def _predict_next_positions(env, agents: Sequence[str], joint_action: Tuple[int, ...]):
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


def _upper_bound(env) -> float:
    """
    Very loose admissible upper bound based on remaining persons.
    If your success reward is ~10000–20000, this is safe.
    """
    remaining = len(env.persons_set)
    return remaining * 20000.0


def _bnb_value(
    env, agents: Sequence[str], depth: int, alpha: float, debug: bool, indent: int
) -> float:
    if depth == 0:
        return 0.0

    ub = _upper_bound(env)
    if ub < alpha:
        if debug and indent <= 2:
            print("  " * indent + f"[B&B] PRUNE: UB={ub:.1f} < alpha={alpha:.1f}")
        return float("-inf")

    best = float("-inf")
    joint_actions = _safe_joint_actions(env, agents)

    if debug and indent <= 2:
        print(
            "  " * indent
            + f"[B&B] depth={depth}, UB={ub:.1f}, alpha={alpha:.1f}, actions={len(joint_actions)}"
        )

    for ja in joint_actions:
        snap = save_env_state(env)

        acts = {a_id: int(a) for a_id, a in zip(agents, ja)}
        obs, rewards, terms, truncs, infos = env.step(acts)

        r = _mean_reward(rewards, agents)
        done = any(terms.values()) or any(truncs.values())

        if done:
            val = r
        else:
            val = r + _bnb_value(env, agents, depth - 1, alpha, debug, indent + 1)

        load_env_state(env, snap)

        if val > best:
            best = val
            if best > alpha:
                alpha = best

    return best


def branch_and_bound_plan(
    env,
    agents: Sequence[str],
    depth: int = 2,
    debug: bool = True,
) -> Dict[str, int]:
    """
    Branch & Bound planner (centralized joint action) without deepcopy().
    """
    if not agents:
        return {}

    joint_actions = _safe_joint_actions(env, agents)

    if debug:
        print(
            f"\n=== Branch&Bound plan: depth={depth}, joint_actions={len(joint_actions)} ==="
        )

    best_ja = None
    best_val = float("-inf")
    alpha = float("-inf")

    for i, ja in enumerate(joint_actions):
        if debug:
            print(f"[B&B] eval {i+1}/{len(joint_actions)} JA={ja} (alpha={alpha:.3f})")

        snap = save_env_state(env)

        acts = {a_id: int(a) for a_id, a in zip(agents, ja)}
        obs, rewards, terms, truncs, infos = env.step(acts)

        r = _mean_reward(rewards, agents)
        done = any(terms.values()) or any(truncs.values())

        if done:
            val = r
        else:
            val = r + _bnb_value(env, agents, depth - 1, alpha, debug, indent=1)

        load_env_state(env, snap)

        if debug:
            print(f"[B&B]   JA={ja} -> Q~{val:.3f}")

        if val > best_val or best_ja is None:
            best_val = val
            best_ja = ja
            alpha = max(alpha, best_val)
            if debug:
                print(f"[B&B]   NEW BEST: JA={best_ja}, Q~{best_val:.3f}")

    return {a_id: int(a) for a_id, a in zip(agents, best_ja)}
