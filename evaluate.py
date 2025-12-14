from __future__ import annotations
import argparse
import datetime
import json
import os
import time
from typing import Dict, List

from DSSE import DroneSwarmSearch
from DSSE.environment.policies.baseline import (
    greedy_pod_policy,
    exploratory_pod_policy,
    random_policy,
)
from DSSE.environment.policies.multi_agent import collaborative_greedy_policy
from DSSE.environment.policies.mcts import mcts_plan_centralized
from DSSE.environment.policies.bb import branch_and_bound_plan
from DSSE.environment.policies.sparse import sparse_sampling_plan


# ======================================= ARG PARSER ==============================================


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark DroneSwarmSearch policies.")

    parser.add_argument("--num-targets", type=int, default=4)
    parser.add_argument("--num-drones", type=int, default=2)
    parser.add_argument("--drone-speed", type=int, default=10)
    parser.add_argument("--grid-size", type=int, default=40)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument(
        "--testing", action="store_true", help="Enable human rendering."
    )
    parser.add_argument("--pod", type=float, default=0.9)
    parser.add_argument("--num-simulations", type=int, default=50)
    parser.add_argument(
        "--policy",
        type=str,
        default="multi",
        choices=["exploratory", "multi", "greedy", "random", "mcts", "sparse", "bb"],
        help="Which policy to benchmark.",
    )
    parser.add_argument("--debug", action="store_true")

    return parser.parse_args()


# ======================================= HELPERS ==============================================
def make_env(args) -> DroneSwarmSearch:
    """Construct the environment using CLI arguments."""
    render_mode = "human" if args.testing else "ansi"

    # center of the grid (works for any grid size)
    center = (args.grid_size // 2, args.grid_size // 2)

    env = DroneSwarmSearch(
        grid_size=args.grid_size,
        render_mode=render_mode,
        render_grid=args.testing,
        render_gradient=args.testing,
        vector=(1, 1),
        timestep_limit=args.max_steps,
        person_amount=args.num_targets,
        dispersion_inc=0.05,
        person_initial_position=center,  # ✅ FIXED
        drone_amount=args.num_drones,
        drone_speed=args.drone_speed,
        probability_of_detection=args.pod,
        pre_render_time=0,
    )
    return env


def select_policy(policy_name: str):
    """Return the policy function based on a string name."""
    name = policy_name.lower()
    if name == "exploratory":
        return exploratory_pod_policy
    if name == "multi":
        return collaborative_greedy_policy
    if name == "greedy":
        return greedy_pod_policy
    if name == "random":
        return random_policy
    if name == "mcts":
        return mcts_plan_centralized
    if name == "sparse":
        return sparse_sampling_plan
    if name == "bb":
        return branch_and_bound_plan
    raise ValueError(f"Unknown policy '{policy_name}'.")


import random
from typing import List, Tuple


def random_unique_positions(
    grid_size: int, k: int, rng: random.Random
) -> List[Tuple[int, int]]:
    """
    Sample k unique (x, y) positions uniformly from the grid.
    """
    if k > grid_size * grid_size:
        raise ValueError("Too many drones for unique positions on this grid.")
    # sample unique cells from flattened indices
    flat = rng.sample(range(grid_size * grid_size), k)
    return [(idx % grid_size, idx // grid_size) for idx in flat]  # (x, y)


def make_reset_options(args, seed: int | None = None) -> dict:
    base = random.Random(seed)

    positions = []
    used = set()
    for drone_idx in range(args.num_drones):
        # different RNG per drone
        rng_d = random.Random(base.randrange(0, 2**31 - 1))

        while True:
            x = rng_d.randrange(args.grid_size)
            y = rng_d.randrange(args.grid_size)
            if (x, y) not in used:
                used.add((x, y))
                positions.append((x, y))
                break

    return {
        "drones_positions": positions,
        "person_pod_multipliers": [1.0] * args.num_targets,
        "vector": (0.3, 0.3),
    }


def save_results_to_file(results: Dict, args, out_dir: str = "./results") -> str:
    os.makedirs(out_dir, exist_ok=True)

    payload = {
        "args": vars(args),
        "results": results,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
    }

    fname = f"{args.policy}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    path = os.path.join(out_dir, fname)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return path


# ===================================== SIMULATION LOOP ========================================


def run_simulations(args) -> Dict:
    env = make_env(args)
    policy_fn = select_policy(args.policy)

    OPTIONS = make_reset_options(args, seed=sim if "sim" in locals() else None)

    if (
        "person_pod_multipliers" in OPTIONS
        and len(OPTIONS["person_pod_multipliers"]) != args.num_targets
    ):
        raise ValueError("Number of POD multipliers must equal number of targets.")

    num_success = 0
    total_saved_across_runs = 0

    ttf_across_runs = []
    ttl_across_successes = []
    leave_events_across_runs = []
    collision_events_across_runs = []
    episode_rewards = []
    total_reward_across_runs = 0.0

    # --- timing containers ---
    sim_durations: List[float] = []

    # total time for this policy across all simulations
    policy_start_time = time.perf_counter()

    seeds = [i for i in range(args.num_simulations)]
    for sim in range(args.num_simulations):
        OPTIONS = make_reset_options(args, seed=seeds[sim])
        if args.debug:
            print(f"\n=== Simulation {sim + 1}/{args.num_simulations} ===")

        sim_start_time = time.perf_counter()

        obs, info = env.reset(options=OPTIONS)
        done = False
        episode_saved = 0
        episode_total_reward = 0.0
        last_infos = None

        while not done:
            # Agents list
            agents_method = getattr(env, "get_agents", None)
            agents = env.get_agents() if callable(agents_method) else env.agents

            if not agents:
                if args.debug:
                    print("No active agents detected; ending episode.")
                break

            # Action selection
            if policy_fn.__name__ == "random_policy":
                actions = policy_fn(obs, agents, env)
            elif (
                policy_fn.__name__ == "mcts_plan_centralized"
                or policy_fn.__name__ == "branch_and_bound_plan"
                or policy_fn.__name__ == "sparse_sampling_plan"
            ):
                actions = policy_fn(env, agents)
            else:
                actions = policy_fn(obs, agents)

            obs, rewards, terminations, truncations, infos = env.step(actions)
            if rewards:
                step_reward = sum(float(r) for r in rewards.values()) / len(rewards)
            else:
                step_reward = 0.0
            episode_total_reward += step_reward
            last_infos = infos
            done = any(terminations.values()) or any(truncations.values())

            remaining = len(env.get_persons())
            episode_saved = args.num_targets - remaining

            if remaining == 0:
                if args.debug:
                    print("All targets saved.")
                num_success += 1
                break

        # ---- per-simulation timing ----
        sim_end_time = time.perf_counter()
        sim_duration = sim_end_time - sim_start_time
        sim_durations.append(sim_duration)

        if args.debug:
            print(f"Simulation runtime: {sim_duration:.4f} seconds")

        # ---- Extract metrics from infos ----
        if last_infos:
            first_agent = next(iter(last_infos.keys()))
            ep_info = last_infos[first_agent]
        else:
            ep_info = {}

        ttf = ep_info.get("time_to_first_detection", args.max_steps)
        if ttf is None:
            ttf = args.max_steps
        ttf_across_runs.append(ttf)

        ttl = ep_info.get("time_to_last_detection", None)
        if ttl is not None:
            ttl_across_successes.append(ttl)

        leaves = ep_info.get("num_leave_grid_events", 0)
        collisions = ep_info.get("num_collision_events", 0)

        leave_events_across_runs.append(leaves)
        collision_events_across_runs.append(collisions)

        total_saved_across_runs += episode_saved

        episode_rewards.append(episode_total_reward)
        total_reward_across_runs += episode_total_reward

    # --- total policy runtime ---
    policy_end_time = time.perf_counter()
    total_runtime = policy_end_time - policy_start_time
    avg_sim_runtime = (
        total_runtime / args.num_simulations if args.num_simulations > 0 else 0.0
    )

    if hasattr(env, "close"):
        env.close()

    # ---- Aggregate statistics ----
    avg_saved = total_saved_across_runs / args.num_simulations
    avg_ttf = sum(ttf_across_runs) / len(ttf_across_runs)
    avg_leaves = sum(leave_events_across_runs) / len(leave_events_across_runs)
    avg_collisions = sum(collision_events_across_runs) / len(
        collision_events_across_runs
    )

    avg_ttl_success = (
        sum(ttl_across_successes) / len(ttl_across_successes)
        if ttl_across_successes
        else None
    )

    min_sim_runtime = min(sim_durations) if sim_durations else 0.0
    max_sim_runtime = max(sim_durations) if sim_durations else 0.0

    avg_episode_reward = (
        total_reward_across_runs / args.num_simulations
        if args.num_simulations > 0
        else 0.0
    )

    # ---- Summary ----
    print("\n================= SUMMARY =================")
    print(f"Policy:                          {args.policy}")
    print(f"Simulations:                     {args.num_simulations}")
    print(f"Grid Size:                       {args.grid_size}")
    print(f"Targets per run:                 {args.num_targets}")
    print(f"Drones:                          {args.num_drones}")
    print(f"Successes (all targets saved):   {num_success}")
    print(f"Avg targets saved per run:       {avg_saved:.2f}")
    print(f"Avg time to first detection:     {avg_ttf:.2f} steps")
    print(
        f"Avg time to last detection:      {avg_ttl_success if avg_ttl_success else 'n/a'}"
    )
    print(f"Avg leave-grid events per run:   {avg_leaves:.2f}")
    print(f"Avg collision events per run:    {avg_collisions:.2f}")
    print("--------------- REWARDS -------------------")
    print(f"Total reward (all simulations):  {total_reward_across_runs:.4f}")
    print(f"Avg reward per simulation:       {avg_episode_reward:.4f}")
    print("--------------- TIMING --------------------")
    print(f"Total runtime (all simulations): {total_runtime:.4f} seconds")
    print(f"Avg runtime per simulation:      {avg_sim_runtime:.4f} seconds")
    print(f"Min simulation runtime:          {min_sim_runtime:.4f} seconds")
    print(f"Max simulation runtime:          {max_sim_runtime:.4f} seconds")

    return {
        # aggregates
        "num_success": num_success,
        "avg_saved": avg_saved,
        "avg_ttf": avg_ttf,
        "avg_ttl_success": avg_ttl_success,
        "avg_leaves": avg_leaves,
        "avg_collisions": avg_collisions,
        "total_runtime": total_runtime,
        "avg_sim_runtime": avg_sim_runtime,
        "min_sim_runtime": min_sim_runtime,
        "max_sim_runtime": max_sim_runtime,
        "total_reward": total_reward_across_runs,
        "avg_episode_reward": avg_episode_reward,
        "ttf_across_runs": ttf_across_runs,
        "ttl_across_successes": ttl_across_successes,
        "leave_events_across_runs": leave_events_across_runs,
        "collision_events_across_runs": collision_events_across_runs,
        "episode_rewards": episode_rewards,
        "sim_durations": sim_durations,
    }


def main():
    args = parse_args()
    results = run_simulations(args)
    out_path = save_results_to_file(results, args, out_dir="./results")
    print(f"\nSaved results to: {out_path}")


if __name__ == "__main__":
    main()
