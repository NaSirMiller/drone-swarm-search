from __future__ import annotations
import argparse
from typing import Dict, List

from DSSE import DroneSwarmSearch
from DSSE.environment.policies.baseline import (
    greedy_pod_policy,
    exploratory_pod_policy,
    random_policy,
)
from DSSE.environment.policies.multi_agent import collaborative_greedy_policy


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
        choices=["exploratory", "multi", "greedy", "random"],
        help="Which policy to benchmark.",
    )
    parser.add_argument("--debug", action="store_true")

    return parser.parse_args()


# ======================================= HELPERS ==============================================


def make_env(args) -> DroneSwarmSearch:
    """Construct the environment using CLI arguments."""
    render_mode = "human" if args.testing else "ansi"

    env = DroneSwarmSearch(
        grid_size=args.grid_size,
        render_mode=render_mode,
        render_grid=args.testing,
        render_gradient=args.testing,
        vector=(1, 1),
        timestep_limit=args.max_steps,
        person_amount=args.num_targets,
        dispersion_inc=0.05,
        person_initial_position=(15, 15),
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
    raise ValueError(f"Unknown policy '{policy_name}'.")


# ===================================== SIMULATION LOOP ========================================


def run_simulations(args) -> Dict:
    env = make_env(args)
    policy_fn = select_policy(args.policy)

    # Reset options (hardcoded for now; can later expose via args)
    OPT = {
        "drones_positions": [(10, 5), (10, 10)],
        "person_pod_multipliers": [0.1, 0.4, 0.5, 1.2],
        "vector": (0.3, 0.3),
    }

    if (
        "person_pod_multipliers" in OPT
        and len(OPT["person_pod_multipliers"]) != args.num_targets
    ):
        raise ValueError("Number of POD multipliers must equal number of targets.")

    num_success = 0
    total_saved_across_runs = 0

    ttf_across_runs = []
    ttl_across_successes = []
    leave_events_across_runs = []
    collision_events_across_runs = []

    for sim in range(args.num_simulations):
        if args.debug:
            print(f"\n=== Simulation {sim + 1}/{args.num_simulations} ===")

        obs, info = env.reset(options=OPT)
        done = False
        episode_saved = 0
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
            else:
                actions = policy_fn(obs, agents)

            obs, rewards, terminations, truncations, infos = env.step(actions)
            last_infos = infos
            done = any(terminations.values()) or any(truncations.values())

            remaining = len(env.get_persons())
            episode_saved = args.num_targets - remaining

            if remaining == 0:
                if args.debug:
                    print("All targets saved.")
                num_success += 1
                break

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

    return {
        "num_success": num_success,
        "avg_saved": avg_saved,
        "avg_ttf": avg_ttf,
        "avg_ttl_success": avg_ttl_success,
        "avg_leaves": avg_leaves,
        "avg_collisions": avg_collisions,
    }


def main():
    args = parse_args()
    run_simulations(args)


if __name__ == "__main__":
    main()
