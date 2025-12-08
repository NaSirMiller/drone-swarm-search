from __future__ import annotations
from typing import Dict, List

from DSSE import DroneSwarmSearch
from DSSE.environment.policies.baseline import greedy_pod_policy, exploratory_pod_policy
from DSSE.environment.policies.multi_agent import collaborative_greedy_policy

# ======================================= CONFIG ==============================================

NUM_TARGETS = 4
NUM_DRONES = 2
DRONE_SPEED = 10
GRID_SIZE = 40
MAX_STEPS = 300
TESTING = True
POD = 0.9
NUM_SIMULATIONS = 50
POLICY = "exploratory"  # "exploratory" | "multi" | "greedy"
DEBUG = True

# Options passed into env.reset()
OPT = {
    "drones_positions": [(10, 5), (10, 10)],
    "person_pod_multipliers": [0.1, 0.4, 0.5, 1.2],
    "vector": (0.3, 0.3),
}


# ======================================= HELPERS ==============================================


def make_env(testing: bool) -> DroneSwarmSearch:
    """Construct the DroneSwarmSearch environment with testing/non-testing settings."""
    render_mode = "human" if testing else "ansi"
    render_grid = testing
    render_gradient = testing

    env = DroneSwarmSearch(
        grid_size=GRID_SIZE,
        render_mode=render_mode,
        render_grid=render_grid,
        render_gradient=render_gradient,
        vector=(1, 1),
        timestep_limit=MAX_STEPS,
        person_amount=NUM_TARGETS,
        dispersion_inc=0.05,
        person_initial_position=(15, 15),
        drone_amount=NUM_DRONES,
        drone_speed=DRONE_SPEED,
        probability_of_detection=POD,
        pre_render_time=0,
    )
    return env


def select_policy(policy_name: str):
    """Return the policy function based on a string name, or raise if invalid."""
    name = policy_name.lower()
    if name == "exploratory":
        return exploratory_pod_policy
    if name == "multi":
        return collaborative_greedy_policy
    if name == "greedy":
        return greedy_pod_policy
    raise ValueError(
        f"Unknown POLICY '{policy_name}'. Use 'exploratory', 'multi', or 'greedy'."
    )


# ===================================== SIMULATION LOOP ========================================


def run_simulations() -> Dict:
    env = make_env(TESTING)
    policy_fn = select_policy(POLICY)

    # Basic sanity check on options
    if (
        "person_pod_multipliers" in OPT
        and len(OPT["person_pod_multipliers"]) != NUM_TARGETS
    ):
        raise ValueError(
            f"person_pod_multipliers length ({len(OPT['person_pod_multipliers'])}) "
            f"must match NUM_TARGETS ({NUM_TARGETS})."
        )

    num_success = 0  # episodes where all targets were saved
    total_saved_across_runs = 0

    ttf_across_runs: List[float] = []  # time to first detection (all runs)
    ttl_across_successes: List[float] = (
        []
    )  # time to last detection (only when all found)
    leave_events_across_runs: List[int] = []  # per-run leave-grid counts
    collision_events_across_runs: List[int] = []  # per-run collision counts

    for sim in range(NUM_SIMULATIONS):
        if DEBUG:
            print(f"\n=== Simulation {sim + 1}/{NUM_SIMULATIONS} ===")

        observations, info = env.reset(options=OPT)
        done = False
        episode_saved = 0
        last_infos = None

        # Main episode loop
        while not done:
            # Some PettingZoo-style envs clear agents at termination; guard against that.
            agents_attr = getattr(env, "get_agents", None)
            if callable(agents_attr):
                agents = env.get_agents()
            else:
                agents = getattr(env, "agents", [])

            if not agents:
                if DEBUG:
                    print("No active agents detected; ending episode.")
                break

            # Compute actions using the selected policy
            actions = policy_fn(observations, agents)

            # Step the environment
            observations, rewards, terminations, truncations, infos = env.step(actions)
            last_infos = infos  # keep latest infos
            done = any(terminations.values()) or any(truncations.values())

            # Remaining persons in the environment
            remaining = len(env.get_persons())
            episode_saved = NUM_TARGETS - remaining

            if remaining == 0:
                if DEBUG:
                    print("All targets saved.")
                num_success += 1
                # env should have set time_to_last_detection at this point
                break

        # ---------- Pull metrics from infos (or env) ----------
        # If we never stepped (weird edge), create a default info
        if last_infos and len(last_infos) > 0:
            # All agents share the same episode-level metrics, so just take one
            first_agent = next(iter(last_infos.keys()))
            ep_info = last_infos[first_agent]
        else:
            ep_info = {}

        # time to first detection
        ttf = ep_info.get("time_to_first_detection", None)
        if ttf is None:
            # Treat no detection as taking MAX_STEPS
            ttf = MAX_STEPS
        ttf_across_runs.append(ttf)

        # time to last detection (only meaningful if all targets were found)
        ttl = ep_info.get("time_to_last_detection", None)
        if ttl is not None:
            ttl_across_successes.append(ttl)

        # leave-grid and collision counts
        leaves = ep_info.get("num_leave_grid_events", 0)
        collisions = ep_info.get("num_collision_events", 0)
        leave_events_across_runs.append(leaves)
        collision_events_across_runs.append(collisions)
        # -----------------------------------------------------

        total_saved_across_runs += episode_saved

    # Close env if it has a close() method (e.g., to shut down pygame)
    if hasattr(env, "close"):
        env.close()

    # Summary statistics
    avg_saved = total_saved_across_runs / NUM_SIMULATIONS
    avg_ttf = sum(ttf_across_runs) / len(ttf_across_runs)
    avg_leaves = sum(leave_events_across_runs) / len(leave_events_across_runs)
    avg_collisions = sum(collision_events_across_runs) / len(
        collision_events_across_runs
    )

    if ttl_across_successes:
        avg_ttl_success = sum(ttl_across_successes) / len(ttl_across_successes)
    else:
        avg_ttl_success = None

    print("\n================= SUMMARY =================")
    print(f"Policy:                          {POLICY}")
    print(f"Simulations:                     {NUM_SIMULATIONS}")
    print(f"Targets per run:                 {NUM_TARGETS}")
    print(f"Successes (all targets saved):   {num_success}")
    print(f"Avg targets saved per run:       {avg_saved:.2f}")
    print(f"Avg time to first detection:     {avg_ttf:.2f} steps (all runs)")
    if avg_ttl_success is not None:
        print(
            f"Avg time to last detection:      {avg_ttl_success:.2f} steps (successful runs)"
        )
    else:
        print("Avg time to last detection:      n/a (no successful runs)")
    print(f"Avg leave-grid events per run:   {avg_leaves:.2f}")
    print(f"Avg collision events per run:    {avg_collisions:.2f}")

    return {
        "num_success": num_success,
        "avg_saved": avg_saved,
        "total_saved": total_saved_across_runs,
        "avg_ttf": avg_ttf,
        "avg_ttl_success": avg_ttl_success,
        "avg_leaves": avg_leaves,
        "avg_collisions": avg_collisions,
    }


def main():
    res = run_simulations()


if __name__ == "__main__":
    main()
