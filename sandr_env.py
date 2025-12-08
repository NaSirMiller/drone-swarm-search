from DSSE import DroneSwarmSearch
from DSSE.environment.policies.baseline import greedy_pod_policy, exploratory_pod_policy
from DSSE.environment.policies.multi_agent import collaborative_greedy_policy

env = DroneSwarmSearch(
    grid_size=40,
    render_mode="human",
    render_grid=True,
    render_gradient=True,
    vector=(1, 1),
    timestep_limit=200,
    person_amount=4,
    dispersion_inc=0.05,
    person_initial_position=(15, 15),
    drone_amount=2,
    drone_speed=10,
    probability_of_detection=0.9,
    pre_render_time=0,
)

opt = {
    "drones_positions": [(10, 5), (10, 10)],
    "person_pod_multipliers": [0.1, 0.4, 0.5, 1.2],
    "vector": (0.3, 0.3),
}
observations, info = env.reset(options=opt)

rewards = 0
done = False
episode = 0
prev_actions = []
while not done:
    actions = exploratory_pod_policy(observations, env.get_agents())
    observations, rewards, terminations, truncations, infos = env.step(actions)
    done = any(terminations.values()) or any(truncations.values())
    episode += 1
print(f"Simulation done at episode={episode}")
print(f"did terminate={done}")
