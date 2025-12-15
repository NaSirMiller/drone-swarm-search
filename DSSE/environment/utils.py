import numpy as np
from DSSE.environment.constants import Actions
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


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


def _choose_scale(max_abs: float):
    """
    Returns (scale_factor, suffix) so values are plotted as value/scale_factor.
    """
    if max_abs >= 1e9:
        return 1e9, " (billions)"
    if max_abs >= 1e6:
        return 1e6, " (millions)"
    if max_abs >= 1e3:
        return 1e3, " (thousands)"
    return 1.0, ""


def _apply_scaled_yaxis(ax, scale: float):
    """
    Make y tick labels show scaled values cleanly (no '1e6' offset text).
    """

    def fmt(y, _):
        y_s = y / scale
        # keep labels readable
        if abs(y_s) >= 100:
            return f"{y_s:,.0f}"
        if abs(y_s) >= 10:
            return f"{y_s:,.1f}"
        return f"{y_s:,.2f}"

    ax.yaxis.set_major_formatter(FuncFormatter(fmt))


def plot_reward_across_t_all_policies(
    policy_results, out_dir, use_symlog=True, linthresh=1000.0
):
    """
    Plot episode reward across simulations for ALL policies (one line per policy).
    Fixes readability by:
      - scaling values (k / M / B)
      - removing scientific offset
      - optional symlog scale to show both near-0 and huge negatives
    """
    # gather all reward values to pick a scale
    all_vals = []
    for res in policy_results.values():
        all_vals.extend(res.get("episode_rewards", []))
    max_abs = float(np.max(np.abs(all_vals))) if all_vals else 0.0
    scale, suffix = _choose_scale(max_abs)

    fig, ax = plt.subplots()

    for policy, res in policy_results.items():
        rewards = res.get("episode_rewards", [])
        if rewards:
            rewards = np.asarray(rewards, dtype=float)
            ax.plot(range(len(rewards)), rewards, label=policy)

    ax.set_xlabel("Simulation")
    ax.set_ylabel(f"Episode Reward{suffix}")
    ax.set_title("Episode Reward Across Simulations (All Policies)")

    # Optional: symlog helps when one policy has giant negative penalties
    if use_symlog and max_abs >= 10 * linthresh:
        ax.set_yscale("symlog", linthresh=linthresh)

    # Apply scaling tick labels (prevents '1e6' offset text)
    _apply_scaled_yaxis(ax, scale)

    ax.legend()

    fig.savefig(
        f"{out_dir}/reward_across_time_all_policies.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_targets_saved_across_t_all_policies(policy_results, out_dir):
    """
    Targets saved per simulation for ALL policies (one line per policy).
    Typically small integers, so no scaling needed.
    """
    fig, ax = plt.subplots()

    for policy, res in policy_results.items():
        saved = res.get("targets_saved_across_runs", [])
        if saved:
            ax.plot(range(len(saved)), saved, label=policy)

    ax.set_xlabel("Simulation")
    ax.set_ylabel("Targets Saved")
    ax.set_title("Targets Saved per Simulation (All Policies)")
    ax.legend()

    fig.savefig(
        f"{out_dir}/targets_saved_across_time_all_policies.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_total_runtime_per_policy(policy_results, out_dir):
    """
    Bar chart of total runtime per policy.
    Scales seconds if needed (rarely necessary, but keeps consistent formatting).
    """
    policies = list(policy_results.keys())
    runtimes = np.array(
        [policy_results[p].get("total_runtime", 0.0) for p in policies], dtype=float
    )

    max_abs = float(np.max(np.abs(runtimes))) if runtimes.size else 0.0
    scale, suffix = _choose_scale(max_abs)

    fig, ax = plt.subplots()
    ax.bar(policies, runtimes)
    ax.set_ylabel(f"Total Runtime (seconds){suffix}")
    ax.set_title("Total Runtime per Policy")

    _apply_scaled_yaxis(ax, scale)

    fig.savefig(
        f"{out_dir}/total_runtime_per_policy.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_ttf_per_policy(policy_results, out_dir):
    """
    Boxplot of TTF per policy. (Usually small enough to be readable.)
    """
    policies = list(policy_results.keys())
    ttf_data = [policy_results[p].get("ttf_across_runs", []) for p in policies]

    fig, ax = plt.subplots()
    ax.boxplot(ttf_data, labels=policies, showfliers=False)
    ax.set_ylabel("Time to First Detection (steps)")
    ax.set_title("TTF Distribution per Policy")

    fig.savefig(
        f"{out_dir}/ttf_per_policy.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_ttl_per_policy(policy_results, out_dir):
    """
    Boxplot of TTL per policy (successful runs only).
    """
    policies = []
    ttl_data = []

    for p, res in policy_results.items():
        arr = res.get("ttl_across_successes", [])
        if arr:
            policies.append(p)
            ttl_data.append(arr)

    fig, ax = plt.subplots()
    ax.boxplot(ttl_data, labels=policies, showfliers=False)
    ax.set_ylabel("Time to Last Detection (steps)")
    ax.set_title("TTL Distribution per Policy (Successful Runs)")

    fig.savefig(
        f"{out_dir}/ttl_per_policy.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_cumulative_reward_per_policy(policy_results, out_dir):
    """
    Bar chart of cumulative reward per policy.
    Scales values to avoid unreadable huge numbers.
    """
    policies = list(policy_results.keys())
    cumulative_rewards = np.array(
        [float(np.sum(policy_results[p].get("episode_rewards", []))) for p in policies],
        dtype=float,
    )

    max_abs = (
        float(np.max(np.abs(cumulative_rewards))) if cumulative_rewards.size else 0.0
    )
    scale, suffix = _choose_scale(max_abs)

    fig, ax = plt.subplots()
    ax.bar(policies, cumulative_rewards)
    ax.set_ylabel(f"Cumulative Reward{suffix}")
    ax.set_title("Cumulative Reward per Policy")

    _apply_scaled_yaxis(ax, scale)

    fig.savefig(
        f"{out_dir}/cumulative_reward_per_policy.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_avg_reward_per_policy(policy_results, out_dir):
    """
    Bar chart of average episode reward per policy.
    Scales values to avoid unreadable huge numbers.
    """
    policies = list(policy_results.keys())
    avg_rewards = np.array(
        [
            (
                float(np.mean(policy_results[p].get("episode_rewards", [])))
                if policy_results[p].get("episode_rewards", [])
                else 0.0
            )
            for p in policies
        ],
        dtype=float,
    )

    max_abs = float(np.max(np.abs(avg_rewards))) if avg_rewards.size else 0.0
    scale, suffix = _choose_scale(max_abs)

    fig, ax = plt.subplots()
    ax.bar(policies, avg_rewards)
    ax.set_ylabel(f"Average Episode Reward{suffix}")
    ax.set_title("Average Reward per Policy")

    _apply_scaled_yaxis(ax, scale)

    fig.savefig(
        f"{out_dir}/avg_reward_per_policy.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)
