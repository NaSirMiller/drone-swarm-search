import argparse
import glob
import json
import os

from DSSE.environment.utils import (
    plot_avg_reward_per_policy,
    plot_cumulative_reward_per_policy,
    plot_reward_across_t_all_policies,
    plot_targets_saved_across_t_all_policies,
    plot_total_runtime_per_policy,
    plot_ttf_per_policy,
    plot_ttl_per_policy,
)


def _policy_name_from_filename(path: str) -> str:
    base = os.path.basename(path)
    return base.split("_", 1)[0]


def load_policy_results(results_dir: str) -> dict:
    """
    Returns:
        policy_results: dict[str, dict] where each value is the saved "results" dict
                       (the same object your plotting helpers expect).
    """
    policy_results: dict = {}

    files = sorted(glob.glob(os.path.join(results_dir, "*.json")))
    if not files:
        raise FileNotFoundError(f"No .json files found in: {results_dir}")

    for fp in files:
        policy = _policy_name_from_filename(fp)
        with open(fp, "r", encoding="utf-8") as f:
            payload = json.load(f)

        res = payload.get("results", payload)

        res.setdefault("episode_rewards", [])
        res.setdefault("ttf_across_runs", [])
        res.setdefault("ttl_across_successes", [])
        res.setdefault("total_runtime", 0.0)

        res.setdefault("targets_saved_across_runs", [])

        policy_results[policy] = res

    return policy_results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="./results")
    ap.add_argument("--plots-dir", default="./plots")
    args = ap.parse_args()

    os.makedirs(args.plots_dir, exist_ok=True)

    policy_results = load_policy_results(args.results_dir)

    # ----- Generate plots -----
    plot_reward_across_t_all_policies(policy_results, args.plots_dir)
    plot_targets_saved_across_t_all_policies(policy_results, args.plots_dir)

    plot_total_runtime_per_policy(policy_results, args.plots_dir)
    plot_ttf_per_policy(policy_results, args.plots_dir)
    plot_ttl_per_policy(policy_results, args.plots_dir)

    plot_cumulative_reward_per_policy(policy_results, args.plots_dir)
    plot_avg_reward_per_policy(policy_results, args.plots_dir)

    print(f"Loaded policies: {', '.join(sorted(policy_results.keys()))}")
    print(f"Saved plots to: {os.path.abspath(args.plots_dir)}")


if __name__ == "__main__":
    main()
