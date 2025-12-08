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


def get_top_k_cells(pod, k):
    # Flatten, sort indices, then unflatten
    flat_idx = np.argpartition(pod.ravel(), -k)[-k:]
    flat_idx = flat_idx[np.argsort(-pod.ravel()[flat_idx])]  # sort desc
    coords = [np.unravel_index(i, pod.shape) for i in flat_idx]
    return coords  # list of (x, y)


def assign_targets_greedy(
    drone_positions, candidate_cells, pod, distance_weight=1.0, pod_weight=5.0
):
    assignments = {}  # drone_index -> (tx, ty)
    remaining_cells = candidate_cells.copy()

    for d_idx, (dx, dy) in enumerate(drone_positions):
        best_cell = None
        best_score = float("inf")

        for cx, cy in remaining_cells:
            dist = abs(dx - cx) + abs(dy - cy)
            score = distance_weight * dist - pod_weight * pod[cx, cy]
            if score < best_score:
                best_score = score
                best_cell = (cx, cy)

        assignments[d_idx] = best_cell
        remaining_cells.remove(best_cell)

    return assignments
