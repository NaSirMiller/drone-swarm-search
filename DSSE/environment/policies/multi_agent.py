import numpy as np

from DSSE.environment.constants import Actions
from DSSE.environment.utils import get_top_k_cells, move_toward

LAST_TARGETS = {}
STUCK_COUNTER: dict = {}


def collaborative_greedy_policy(
    obs,
    agents,
    repulsion_weight: float = 15.0,
    repulsion_radius: int = 2,
    search_threshold: float = 0.0,
    k_factor: int = 4,
    epsilon_local_explore: float = 0.3,
):
    """
    Collaborative regional policy:
    - Applies a global repulsion field on the shared POD map based on drone positions.
    - Finds the global top-K highest probability cells (from the repulsed map).
    - Partitions those top-K cells among the drones (each drone gets its own subset).
    - For each drone, focuses on the hottest cell in its subset and explores that
      cell and its immediate neighbors (center + 4-neighborhood).
    - If the drone is already at the best local cell:
        * With probability (1 - epsilon_local_explore) and if POD >= search_threshold,
          it SEARCHes.
        * With probability epsilon_local_explore (or if POD < search_threshold),
          it moves to a high-POD neighbor instead (local exploration).
    """
    actions = {}

    if not agents:
        return actions

    # 1) Extract joint info
    drone_positions = [obs[agent][0] for agent in agents]
    # All drones see the same POD matrix
    pod = next(iter(obs.values()))[1]
    pod_mod = pod.copy()

    # 2) Apply repulsion around all drones on the global POD map
    if repulsion_weight is not None and repulsion_weight > 0:
        for ox, oy in drone_positions:
            for dx in range(-repulsion_radius, repulsion_radius + 1):
                for dy in range(-repulsion_radius, repulsion_radius + 1):
                    cx, cy = ox + dx, oy + dy
                    if 0 <= cx < pod_mod.shape[0] and 0 <= cy < pod_mod.shape[1]:
                        pod_mod[cx, cy] -= repulsion_weight / (abs(dx) + abs(dy) + 1)

        pod_mod = np.clip(pod_mod, 0.0, 1.0)

    # 3) Get global top-K cells from the modified POD map
    num_agents = len(agents)
    K = max(num_agents * k_factor, num_agents)
    candidate_cells = get_top_k_cells(pod_mod, K)  # list of (x, y) cells, highest first

    if not candidate_cells:
        # Fallback: everyone just goes to global argmax if something went wrong
        tx, ty = np.unravel_index(np.argmax(pod_mod), pod_mod.shape)
        for agent, (ax, ay) in zip(agents, drone_positions):
            actions[agent] = move_toward((ax, ay), (tx, ty))
        return actions

    # 4) Partition the candidate cells among the drones (round-robin)
    clusters = {agent: [] for agent in agents}
    for i, cell in enumerate(candidate_cells):
        agent = agents[i % num_agents]
        clusters[agent].append(cell)

    # 5) For each agent, pick the hottest cell in its subset,
    #    then explore that cell and its neighbors.
    H, W = pod_mod.shape

    for idx, agent in enumerate(agents):
        ax, ay = drone_positions[idx]
        my_cells = clusters.get(agent, [])

        if not my_cells:
            # Fallback: no assigned cells (e.g., more drones than K)
            tx, ty = np.unravel_index(np.argmax(pod_mod), pod_mod.shape)
            actions[agent] = move_toward((ax, ay), (tx, ty))
            continue

        # Pick the highest-probability cell in this agent's partition
        hot_x, hot_y = max(my_cells, key=lambda c: pod_mod[c[0], c[1]])

        # Build a local neighborhood around this hot cell (center + 4-neighbors)
        local_cells = [(hot_x, hot_y)]
        if hot_x + 1 < H:
            local_cells.append((hot_x + 1, hot_y))
        if hot_x - 1 >= 0:
            local_cells.append((hot_x - 1, hot_y))
        if hot_y + 1 < W:
            local_cells.append((hot_x, hot_y + 1))
        if hot_y - 1 >= 0:
            local_cells.append((hot_x, hot_y - 1))

        # Among this local neighborhood, pick the best POD cell
        best_x, best_y = max(local_cells, key=lambda c: pod_mod[c[0], c[1]])
        best_val = pod_mod[best_x, best_y]

        # ---- Behavior when at the best local cell ----
        if (ax, ay) == (best_x, best_y):
            # Decide whether to SEARCH or locally explore neighbors
            # If POD is high and we don't explore this time: SEARCH
            if (best_val >= search_threshold) and (
                np.random.rand() > epsilon_local_explore
            ):
                actions[agent] = Actions.SEARCH.value
            else:
                # Explore: move toward the best neighbor (excluding current cell)
                neighbor_candidates = [c for c in local_cells if c != (ax, ay)]
                if neighbor_candidates:
                    nx, ny = max(
                        neighbor_candidates,
                        key=lambda c: pod_mod[c[0], c[1]],
                    )
                    actions[agent] = move_toward((ax, ay), (nx, ny))
                else:
                    # No valid neighbors (edge case) → SEARCH
                    actions[agent] = Actions.SEARCH.value

        # ---- Behavior when not at the best local cell ----
        else:
            actions[agent] = move_toward((ax, ay), (best_x, best_y))

    return actions
