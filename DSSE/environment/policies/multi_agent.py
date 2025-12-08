import numpy as np

from DSSE.environment.constants import Actions
from DSSE.environment.utils import assign_targets_greedy, get_top_k_cells, move_toward


def collaborative_greedy_policy(obs, agents, repulsion_weight=0):
    actions = {}

    # Extract joint info once
    drone_positions = [obs[agent][0] for agent in agents]
    pod = next(iter(obs.values()))[1]  # same for all

    pod_mod = pod.copy()

    # 1) Get candidate targets
    K = max(len(agents) * 2, len(agents))
    candidate_cells = get_top_k_cells(pod_mod, K)

    # 2) Assign targets collaboratively
    assignments = assign_targets_greedy(drone_positions, candidate_cells, pod_mod)

    # 3) For each agent, move toward its assigned target
    for idx, agent in enumerate(agents):
        ax, ay = drone_positions[idx]
        tx, ty = assignments[idx]

        # If already at target, SEARCH, else move toward
        if (ax, ay) == (tx, ty):
            actions[agent] = Actions.SEARCH.value
        else:
            actions[agent] = move_toward((ax, ay), (tx, ty))

    return actions
