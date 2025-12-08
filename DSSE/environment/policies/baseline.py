import numpy as np

from DSSE.environment.constants import Actions


def move_toward(curr: tuple[int, int], target: tuple[int, int]):
    """
    Simple mapping from (curr_x,curr_y) -> (target_x,target_y) into one discrete action.
    Prioritizes the larger delta (Manhattan). If already at target, returns SEARCH.
    """
    cx, cy = curr
    tx, ty = target
    dx = tx - cx
    dy = ty - cy

    if dx == 0 and dy == 0:
        print("searching...")
        return Actions.SEARCH.value

    # prefer horizontal when abs(dx) > abs(dy), else vertical
    if abs(dx) > abs(dy):
        print("moving horizontally...")
        return Actions.RIGHT.value if dx > 0 else Actions.LEFT.value
    else:
        print("moving vertically...")
        return Actions.DOWN.value if dy > 0 else Actions.UP.value


def greedy_pod_policy(obs, agents, repulsion_weight=5):
    actions = {}

    # Gather all drone positions for repulsion
    drone_positions = {agent: obs[agent][0] for agent in agents}  # 0 = position

    for agent in agents:
        print(f"================ AGENT {agent} ==================")
        position, pod = obs[agent]  # unpack tuple
        ax, ay = position

        # Copy POD map so we can modify it
        pod_mod = pod.copy()

        # Apply repulsion from other drones
        for other_agent, (ox, oy) in drone_positions.items():
            if other_agent == agent:
                continue

            # Strongly penalize nearby cells
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    cx, cy = ox + dx, oy + dy
                    if 0 <= cx < pod.shape[0] and 0 <= cy < pod.shape[1]:
                        pod_mod[cx, cy] -= repulsion_weight / (abs(dx) + abs(dy) + 1)

        # Target is now the highest value in the modified map
        tx, ty = np.unravel_index(np.argmax(pod_mod), pod_mod.shape)

        # Move toward that target
        actions[agent] = move_toward((ax, ay), (tx, ty))

    return actions


def exploratory_pod_policy(
    obs,
    agents,
    repulsion_weight=5,
    search_threshold=0.0,
    epsilon=0.15,
):
    """
    Greedy-but-exploratory policy:
    - Applies repulsion from other drones.
    - Follows local POD gradient instead of global argmax.
    - Searches only at local maxima above `search_threshold`.
    - With prob `epsilon`, takes a random move for exploration.
    """
    actions = {}

    # Gather all drone positions for repulsion
    drone_positions = {agent: obs[agent][0] for agent in agents}  # 0 = position

    for agent in agents:
        position, pod = obs[agent]
        ax, ay = position

        # Copy POD map so we can modify it
        pod_mod = pod.copy()

        # --- Repulsion from other drones (same as your baseline) ---
        for other_agent, (ox, oy) in drone_positions.items():
            if other_agent == agent:
                continue

            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    cx, cy = ox + dx, oy + dy
                    if 0 <= cx < pod.shape[0] and 0 <= cy < pod.shape[1]:
                        pod_mod[cx, cy] -= repulsion_weight / (abs(dx) + abs(dy) + 1)

        pod_mod = np.clip(pod_mod, 0.0, 1.0)

        # --- Local neighborhood (center + 4-neighbors) ---
        neighbors = []
        # center
        neighbors.append((ax, ay))

        # 4-connected neighbors (check bounds)
        if ax + 1 < pod_mod.shape[0]:
            neighbors.append((ax + 1, ay))
        if ax - 1 >= 0:
            neighbors.append((ax - 1, ay))
        if ay + 1 < pod_mod.shape[1]:
            neighbors.append((ax, ay + 1))
        if ay - 1 >= 0:
            neighbors.append((ax, ay - 1))

        # --- epsilon-greedy exploration: sometimes just move randomly ---
        if len(neighbors) > 1 and np.random.rand() < epsilon:
            # Pick a random *neighbor different from current cell* to move towards
            candidate_neighbors = [(x, y) for (x, y) in neighbors if (x, y) != (ax, ay)]
            nx, ny = candidate_neighbors[np.random.randint(len(candidate_neighbors))]
            actions[agent] = move_toward((ax, ay), (nx, ny))
            continue

        # --- Greedy over local neighborhood ---
        center_val = pod_mod[ax, ay]

        best_val = -np.inf
        best_pos = (ax, ay)
        for nx, ny in neighbors:
            val = pod_mod[nx, ny]
            if val > best_val:
                best_val = val
                best_pos = (nx, ny)

        # If current cell is a local maximum and high enough, SEARCH
        if best_pos == (ax, ay) and center_val >= search_threshold:
            actions[agent] = Actions.SEARCH.value
        else:
            # Move toward neighbor with highest POD
            actions[agent] = move_toward((ax, ay), best_pos)

    return actions


def random_policy(obs, agents, env):
    actions = {}
    for agent in agents:
        actions[agent] = env.action_space(agent).sample()
    return actions
