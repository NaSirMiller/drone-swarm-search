import numpy as np
from DSSE.environment.constants import Actions
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def move_toward(curr: tuple[int, int], target: tuple[int, int]):
    """
    Map (curr_x, curr_y) -> (target_x, target_y) into a discrete action.

    - If already at target: SEARCH
    - Otherwise: move along the dominant axis
    - If |dx| == |dy| (tie), break ties randomly to avoid deterministic
      up/down or left/right oscillations.
    """
    cx, cy = curr
    tx, ty = target
    dx = tx - cx
    dy = ty - cy

    # Already at target: search
    if dx == 0 and dy == 0:
        return Actions.SEARCH.value

    adx = abs(dx)
    ady = abs(dy)

    # Prefer horizontal if strictly larger
    if adx > ady:
        return Actions.RIGHT.value if dx > 0 else Actions.LEFT.value

    # Prefer vertical if strictly larger
    if ady > adx:
        return Actions.DOWN.value if dy > 0 else Actions.UP.value

    # Tie: |dx| == |dy| and both non-zero -> break tie randomly
    if np.random.rand() < 0.5:
        return Actions.RIGHT.value if dx > 0 else Actions.LEFT.value
    else:
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


def plot_reward_across_t_all_policies(policy_results, out_dir):
    """
    Plot episode reward across simulations for ALL policies (one line per policy).
    Expects: policy_results[policy]["episode_rewards"] -> list[float]
    """
    plt.figure()

    for policy, res in policy_results.items():
        rewards = res.get("episode_rewards", [])
        if rewards:
            plt.plot(range(len(rewards)), rewards, label=policy)

    plt.xlabel("Simulation")
    plt.ylabel("Episode Reward")
    plt.title("Episode Reward Across Simulations (All Policies)")
    plt.legend()

    plt.savefig(
        f"{out_dir}/reward_across_time_all_policies.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def plot_targets_saved_across_t_all_policies(policy_results, out_dir):
    """
    Plot targets saved per simulation for ALL policies (one line per policy).
    Expects: policy_results[policy]["targets_saved_across_runs"] -> list[int|float]
    """
    plt.figure()

    for policy, res in policy_results.items():
        saved = res.get("targets_saved_across_runs", [])
        if saved:
            plt.plot(range(len(saved)), saved, label=policy)

    plt.xlabel("Simulation")
    plt.ylabel("Targets Saved")
    plt.title("Targets Saved per Simulation (All Policies)")
    plt.legend()

    plt.savefig(
        f"{out_dir}/targets_saved_across_time_all_policies.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def plot_total_runtime_per_policy(policy_results, out_dir):
    """
    Bar chart of total runtime per policy.
    Expects: policy_results[policy]["total_runtime"] -> float
    """
    policies = list(policy_results.keys())
    runtimes = [policy_results[p].get("total_runtime", 0.0) for p in policies]

    plt.figure()
    plt.bar(policies, runtimes)
    plt.ylabel("Total Runtime (seconds)")
    plt.title("Total Runtime per Policy")

    plt.savefig(
        f"{out_dir}/total_runtime_per_policy.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def plot_ttf_per_policy(policy_results, out_dir):
    """
    Boxplot of TTF per policy.
    Expects: policy_results[policy]["ttf_across_runs"] -> list[int|float]
    """
    policies = list(policy_results.keys())
    ttf_data = [policy_results[p].get("ttf_across_runs", []) for p in policies]

    plt.figure()
    plt.boxplot(ttf_data, labels=policies, showfliers=False)
    plt.ylabel("Time to First Detection (steps)")
    plt.title("TTF Distribution per Policy")

    plt.savefig(
        f"{out_dir}/ttf_per_policy.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def plot_ttl_per_policy(policy_results, out_dir):
    """
    Boxplot of TTL per policy (successful runs only).
    Expects: policy_results[policy]["ttl_across_successes"] -> list[int|float]
    """
    policies = []
    ttl_data = []

    for p, res in policy_results.items():
        arr = res.get("ttl_across_successes", [])
        if arr:  # only include policies with at least one successful TTL recorded
            policies.append(p)
            ttl_data.append(arr)

    plt.figure()
    plt.boxplot(ttl_data, labels=policies, showfliers=False)
    plt.ylabel("Time to Last Detection (steps)")
    plt.title("TTL Distribution per Policy (Successful Runs)")

    plt.savefig(
        f"{out_dir}/ttl_per_policy.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def plot_cumulative_reward_per_policy(policy_results, out_dir):
    """
    Bar chart of cumulative reward per policy.
    Expects: policy_results[policy]["episode_rewards"] -> list[float]
    """
    policies = list(policy_results.keys())
    cumulative_rewards = [
        float(np.sum(policy_results[p].get("episode_rewards", []))) for p in policies
    ]

    plt.figure()
    plt.bar(policies, cumulative_rewards)
    plt.ylabel("Cumulative Reward")
    plt.title("Cumulative Reward per Policy")

    plt.savefig(
        f"{out_dir}/cumulative_reward_per_policy.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def plot_avg_reward_per_policy(policy_results, out_dir):
    """
    Bar chart of average episode reward per policy.
    Expects: policy_results[policy]["episode_rewards"] -> list[float]
    """
    policies = list(policy_results.keys())
    avg_rewards = []
    for p in policies:
        rewards = policy_results[p].get("episode_rewards", [])
        avg_rewards.append(float(np.mean(rewards)) if rewards else 0.0)

    plt.figure()
    plt.bar(policies, avg_rewards)
    plt.ylabel("Average Episode Reward")
    plt.title("Average Reward per Policy")

    plt.savefig(
        f"{out_dir}/avg_reward_per_policy.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
