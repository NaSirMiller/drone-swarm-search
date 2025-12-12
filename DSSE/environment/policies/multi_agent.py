from collections import defaultdict
import numpy as np

from DSSE.environment.constants import Actions
from DSSE.environment.utils import assign_targets_greedy, get_top_k_cells, move_toward

LAST_TARGETS = {}
STUCK_COUNTER: dict = {}


def _next_pos_from_action(pos, action_value: int) -> tuple[int, int]:
    """
    Given a (x, y) grid position and an action value, return the next (x, y).
    SEARCH means stay in place.
    """
    x, y = pos

    if action_value == Actions.UP.value:
        return (x, y - 1)
    if action_value == Actions.DOWN.value:
        return (x, y + 1)
    if action_value == Actions.LEFT.value:
        return (x - 1, y)
    if action_value == Actions.RIGHT.value:
        return (x + 1, y)

    # SEARCH or anything else: stay put
    return (x, y)


# def collaborative_greedy_policy(
#     obs,
#     agents,
#     repulsion_weight: float = 25.0,
#     repulsion_radius: int = 3,
#     hysteresis_delta: float = 0.02,
#     stuck_threshold: int = 3,
#     epsilon_explore: float = 0.15,
# ):
#     """
#     Collaborative greedy policy:
#     - Applies a global repulsion field on the shared POD map based on drone positions.
#     - Picks top-K high-POD cells from the repulsed map.
#     - Greedily assigns targets to drones to minimize total travel cost.
#     - Uses hysteresis: each drone keeps its previous target unless a new one
#       is significantly better in POD by `hysteresis_delta`.
#     - Adds exploration:
#         * If an agent SEARCHes at the same target for too many consecutive
#           steps (stuck_threshold), it moves toward a high-POD neighbor instead.
#         * With probability epsilon_explore, an agent moving toward its target
#           will instead move toward a random neighbor.
#     - Adds action-level collision avoidance:
#         * After choosing actions, simulate next positions.
#         * If multiple agents would move into the same cell, only one moves;
#           the others SEARCH (wait) to avoid collisions.
#     """
#     actions = {}

#     # 1) Extract joint info
#     drone_positions = [obs[agent][0] for agent in agents]
#     pod = next(iter(obs.values()))[1]  # assumed same for all agents
#     pod_mod = pod.copy()

#     # 2) Apply repulsion around all drones on the global POD map
#     if repulsion_weight is not None and repulsion_weight > 0:
#         for ox, oy in drone_positions:
#             for dx in range(-repulsion_radius, repulsion_radius + 1):
#                 for dy in range(-repulsion_radius, repulsion_radius + 1):
#                     cx, cy = ox + dx, oy + dy
#                     if 0 <= cx < pod_mod.shape[0] and 0 <= cy < pod_mod.shape[1]:
#                         pod_mod[cx, cy] -= repulsion_weight / (abs(dx) + abs(dy) + 1)

#         pod_mod = np.clip(pod_mod, 0.0, 1.0)

#     # 3) Get candidate targets from the *modified* POD map
#     K = max(len(agents) * 2, len(agents))
#     candidate_cells = get_top_k_cells(pod_mod, K)  # list of (x, y) cells

#     # 4) Assign targets collaboratively (using modified POD map)
#     #    assignments: list of (tx, ty) aligned with drone_positions
#     assignments = assign_targets_greedy(drone_positions, candidate_cells, pod_mod)

#     # 5) Apply hysteresis per agent: keep previous target if new one
#     #    is not significantly better in POD.
#     global LAST_TARGETS, STUCK_COUNTER
#     chosen_targets: list[tuple[int, int]] = []

#     for idx, agent in enumerate(agents):
#         new_target = assignments[idx]
#         old_target = LAST_TARGETS.get(agent)

#         # If we have a previous target, compare POD values
#         if old_target is not None:
#             ox, oy = old_target
#             nx, ny = new_target

#             # Safety clamp in case something weird got stored
#             if not (0 <= ox < pod_mod.shape[0] and 0 <= oy < pod_mod.shape[1]):
#                 old_val = -np.inf
#             else:
#                 old_val = pod_mod[ox, oy]

#             if not (0 <= nx < pod_mod.shape[0] and 0 <= ny < pod_mod.shape[1]):
#                 new_val = -np.inf
#             else:
#                 new_val = pod_mod[nx, ny]

#             # Only switch targets if new target is significantly better
#             if new_val < old_val + hysteresis_delta:
#                 # Keep old target
#                 chosen_target = old_target
#             else:
#                 chosen_target = new_target
#         else:
#             # No previous target: use the new one
#             chosen_target = new_target

#         LAST_TARGETS[agent] = chosen_target
#         # Reset stuck counter when target changes
#         if old_target is None or old_target != chosen_target:
#             STUCK_COUNTER[agent] = 0

#         chosen_targets.append(chosen_target)

#     # 6) For each agent, choose an intended action (with exploration / stuck logic)
#     intended_actions: dict = {}
#     next_positions: dict = {}

#     for idx, agent in enumerate(agents):
#         ax, ay = drone_positions[idx]
#         tx, ty = chosen_targets[idx]

#         if (ax, ay) == (tx, ty):
#             # Agent is at its target; might SEARCH or EXPLORE

#             # Increase stuck counter
#             STUCK_COUNTER[agent] = STUCK_COUNTER.get(agent, 0) + 1

#             if STUCK_COUNTER[agent] >= stuck_threshold:
#                 # --- Exploration: move toward the best neighbor cell instead of SEARCH ---
#                 neighbors: list[tuple[int, int]] = []
#                 if ax + 1 < pod_mod.shape[0]:
#                     neighbors.append((ax + 1, ay))
#                 if ax - 1 >= 0:
#                     neighbors.append((ax - 1, ay))
#                 if ay + 1 < pod_mod.shape[1]:
#                     neighbors.append((ax, ay + 1))
#                 if ay - 1 >= 0:
#                     neighbors.append((ax, ay - 1))

#                 if neighbors:
#                     # Pick neighbor with highest modified POD
#                     best_n = None
#                     best_val = -np.inf
#                     for nx, ny in neighbors:
#                         val = pod_mod[nx, ny]
#                         if val > best_val:
#                             best_val = val
#                             best_n = (nx, ny)

#                     ex_tx, ex_ty = best_n
#                     act = move_toward((ax, ay), (ex_tx, ex_ty))

#                     # Update target to this new neighbor and reset stuck counter
#                     LAST_TARGETS[agent] = (ex_tx, ex_ty)
#                     STUCK_COUNTER[agent] = 0
#                 else:
#                     # No valid neighbors (edge weirdness) → fallback to SEARCH
#                     act = Actions.SEARCH.value
#             else:
#                 # Not yet stuck → SEARCH at target
#                 act = Actions.SEARCH.value
#         else:
#             # Not at target → move toward target, with small epsilon exploration
#             if np.random.rand() < epsilon_explore:
#                 # Explore: move toward a random 4-neighbor instead of precise target
#                 neighbors: list[tuple[int, int]] = []
#                 if ax + 1 < pod_mod.shape[0]:
#                     neighbors.append((ax + 1, ay))
#                 if ax - 1 >= 0:
#                     neighbors.append((ax - 1, ay))
#                 if ay + 1 < pod_mod.shape[1]:
#                     neighbors.append((ax, ay + 1))
#                 if ay - 1 >= 0:
#                     neighbors.append((ax, ay - 1))

#                 if neighbors:
#                     rand_nx, rand_ny = neighbors[np.random.randint(len(neighbors))]
#                     act = move_toward((ax, ay), (rand_nx, rand_ny))
#                 else:
#                     # Fallback if no neighbors (shouldn't happen in normal grids)
#                     act = move_toward((ax, ay), (tx, ty))
#             else:
#                 # Greedy move toward target
#                 act = move_toward((ax, ay), (tx, ty))

#             # Reset stuck counter since we are moving
#             STUCK_COUNTER[agent] = 0

#         intended_actions[agent] = act
#         next_positions[agent] = _next_pos_from_action((ax, ay), act)

#     # 7) Collision avoidance: if multiple agents want the same next position,
#     #    let only one move; others SEARCH (wait in place).
#     pos_to_agents: dict[tuple[int, int], list] = defaultdict(list)
#     for agent in agents:
#         pos_to_agents[next_positions[agent]].append(agent)

#     for pos, group in pos_to_agents.items():
#         if len(group) > 1:
#             # Conflict: multiple agents want to occupy `pos`
#             # Keep the first agent's action, force others to SEARCH (stay put).
#             keeper = group[0]
#             for loser in group[1:]:
#                 intended_actions[loser] = Actions.SEARCH.value
#                 # Their next position becomes their current position
#                 idx = agents.index(loser)
#                 curr_pos = drone_positions[idx]
#                 next_positions[loser] = curr_pos

#     # Final actions
#     for agent in agents:
#         actions[agent] = intended_actions[agent]

#     return actions


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
