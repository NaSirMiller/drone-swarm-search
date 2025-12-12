from random import random, randint
import functools
import numpy as np
from gymnasium.spaces import MultiDiscrete, Discrete, Box, Tuple
from .constants import RED, GREEN, Actions, Reward
from .entities.person import Person
from .env_base import DroneSwarmSearchBase
from .simulation.dynamic_probability import ProbabilityMatrix


class DroneSwarmSearch(DroneSwarmSearchBase):
    """
    PettingZoo based environment for SAR missions using drones.
    """

    possible_actions = {action for action in Actions}
    metadata = {
        "name": "DroneSwarmSearchV0",
    }

    # -------------------------------------------------------------------------
    # REWARD SCHEME (paper)
    # -------------------------------------------------------------------------
    # Move (default)                            -> +1
    # Move and leave grid                       -> -100000
    # Move and collide with other agents        -> -100000
    # Search cell with prob < 1%                -> -100
    # Search cell with prob >= 1% (no person)   -> prob(cell) * 10000
    # Search cell containing shipwrecked person -> 10000 + 10000 * (1 - timestep/timestep_limit)
    # Any action beyond timestep_limit          -> -100000
    reward_scheme = Reward(
        default=1.0,
        leave_grid=-100000.0,
        exceed_timestep=-100000.0,
        drones_collision=-100000.0,
        search_cell=-100.0,  # used for low-probability searches
        search_and_find=10000.0,  # base term for successful detection
    )

    def __init__(
        self,
        grid_size=20,
        render_mode="ansi",
        render_grid=False,
        render_gradient=True,
        vector=(1.1, 1),
        dispersion_inc=0.1,
        dispersion_start=0.5,
        timestep_limit=100,
        person_amount=1,
        person_initial_position=(0, 0),
        drone_amount=1,
        drone_speed=10,
        probability_of_detection=1.0,
        pre_render_time=0,
        grid_cell_size=130,
        fps=5,
        use_global_reward=False,
    ):
        if person_amount <= 0:
            raise ValueError("The number of persons must be greater than 0.")
        self.person_amount = person_amount

        super().__init__(
            grid_size=grid_size,
            render_mode=render_mode,
            render_grid=render_grid,
            render_gradient=render_gradient,
            timestep_limit=timestep_limit,
            drone_amount=drone_amount,
            drone_speed=drone_speed,
            probability_of_detection=probability_of_detection,
            grid_cell_size=grid_cell_size,
            render_fps=fps,
        )

        self.pre_render_steps = round(
            (pre_render_time * 60)
            / (self.calculate_simulation_time_step(drone_speed, self.cell_size))
        )
        # print(f"Pre render time: {pre_render_time} minutes")
        # print(f"Pre render steps: {self.pre_render_steps}")

        # Prob matrix
        self.probability_matrix = None
        self.dispersion_inc = dispersion_inc
        self.dispersion_start = dispersion_start
        self.vector = vector
        self.disaster_position = person_initial_position

        # Person initialization
        self.person_initial_position = person_initial_position
        self.persons_set = self.create_persons_set()

        # Initializing render
        self.rewards_sum = {a: 0 for a in self.possible_agents}
        self.rewards_sum["total"] = 0

        # Define the action and observation spaces for compatibility with RL libraries
        self.action_spaces = {
            agent: self.action_space(agent) for agent in self.possible_agents
        }
        self.observation_spaces = {
            agent: self.observation_space(agent) for agent in self.possible_agents
        }
        self.use_global_reward = use_global_reward

    def create_persons_set(self) -> set[Person]:
        persons_set = set()
        position = self.create_random_positions_person(
            self.person_initial_position, self.person_amount
        )

        for i in range(self.person_amount):
            person = Person(
                index=i,
                initial_position=position[i],
                grid_size=self.grid_size,
            )
            person.calculate_movement_vector(self.vector)
            persons_set.add(person)

            if not self.is_valid_position(person.initial_position):
                raise ValueError("Person initial position is out of the matrix")

        return persons_set

    def create_random_positions_person(
        self, central_position: tuple[int, int], amount: int, max_distance: int = 2
    ) -> list[tuple[int, int]]:
        if not self.is_valid_position(central_position):
            raise ValueError("Central position is out of the matrix")

        max_distance_range = (max_distance * 2 + 1) ** 2

        if amount > max_distance_range:
            raise ValueError(
                "There are more persons than grid spots. Reduce number of persons or increase grid size."
            )

        unique_random_positions = {central_position}
        while len(unique_random_positions) < amount:
            dx = randint(-max_distance, max_distance)
            dy = randint(-max_distance, max_distance)

            # Checking to avoid including the central position or duplicate positions.
            if (dx, dy) != (0, 0):
                new_position = (central_position[0] + dx, central_position[1] + dy)
                if self.is_valid_position(new_position):
                    unique_random_positions.add(new_position)

        return list(unique_random_positions)

    def render(self):
        self.pygame_renderer.render_map()
        self.pygame_renderer.render_entities(self.persons_set)
        self.pygame_renderer.render_entities(self.agents_positions)
        self.pygame_renderer.refresh_screen()

    def reset(
        self,
        seed=None,
        options=None,
    ):
        vector = options.get("vector") if options else None
        self.vector = vector if vector else self.vector

        self.persons_set = self.create_persons_set()
        for person in self.persons_set:
            person.reset_position()
            person.update_time_step_relation(self.time_step_relation, self.cell_size)

        pod_multiplier = options.get("person_pod_multipliers") if options else None

        if pod_multiplier is not None:
            self.raise_if_unvalid_mult(pod_multiplier)
            for person, mult in zip(self.persons_set, pod_multiplier):
                person.set_mult(mult)

        self.probability_matrix = ProbabilityMatrix(
            40,
            self.dispersion_start,
            self.dispersion_inc,
            self.vector,
            [
                self.disaster_position[1],
                self.disaster_position[0],
            ],
            self.grid_size,
        )

        self.probability_matrix.update_time_step_relation(
            self.time_step_relation, self.cell_size
        )

        observations, infos = super().reset(seed=seed, options=options)
        self.rewards_sum = {a: 0 for a in self.agents}
        self.rewards_sum["total"] = 0

        # episode level metrics
        self.time_to_first_detection = None
        self.time_to_last_detection = None
        self.num_leave_grid_events = 0
        self.num_collision_events = 0

        return observations, infos

    def raise_if_unvalid_mult(self, individual_multiplication: list[int]) -> bool:
        if len(individual_multiplication) != len(self.persons_set):
            raise ValueError(
                "The number of multipliers must be equal to the number of persons."
            )

        for mult in individual_multiplication:
            if not isinstance(mult, (int, float)) or mult < 0:
                raise ValueError("The multiplier must be a positive number.")

    def pre_search_simulate(self):
        for _ in range(self.pre_render_steps):
            self.create_observations()
            if self.render_mode == "human":
                self.render()

    def create_observations(self):
        observations = {}

        for person in self.persons_set:
            movement_map = self.build_movement_matrix(person)
            person.step(movement_map)

        self.probability_matrix.step()

        probability_matrix = self.probability_matrix.get_matrix()
        for idx, agent in enumerate(self.agents):
            observation = (
                self.agents_positions[idx],
                probability_matrix,
            )
            observations[agent] = observation

        return observations

    def step(self, actions):
        """
        Returns a tuple with (observations, rewards, terminations, truncations, infos)
        """
        if not self._was_reset:
            raise ValueError("Please reset the env before interacting with it")

        terminations = {a: False for a in self.agents}
        rewards = {a: 0.0 for a in self.agents}
        truncations = {a: False for a in self.agents}
        person_found = False
        collision_this_step = False

        # Probability map at *current* timestep (before we advance it in create_observations)
        prob_matrix = self.probability_matrix.get_matrix()

        for idx, agent in enumerate(self.agents):
            if agent not in actions:
                raise ValueError("Missing action for " + agent)

            drone_action = actions[agent]
            if drone_action not in self.action_space(agent):
                raise ValueError("Invalid action for " + agent)

            # Check truncation conditions (paper: any action after timestep_limit -> -100000)
            if self.timestep >= self.timestep_limit:
                rewards[agent] = self.reward_scheme.exceed_timestep
                truncations[agent] = True
                terminations[agent] = True
                continue

            drone_x, drone_y = self.agents_positions[idx]
            is_searching = drone_action == Actions.SEARCH.value

            # ------------------ MOVEMENT / LEAVE-GRID / COLLISIONS ------------------
            if not is_searching:
                new_position = self.move_drone((drone_x, drone_y), drone_action)
                if not self.is_valid_position(new_position):
                    # Leaving the grid: large negative reward
                    rewards[agent] = self.reward_scheme.leave_grid
                    self.num_leave_grid_events += 1
                else:
                    # Valid movement with default +1 reward *unless* a collision happens
                    self.agents_positions[idx] = new_position
                    rewards[agent] = self.reward_scheme.default

                    # Check for collisions
                    for j, other_agent in enumerate(self.agents):
                        if (
                            j != idx
                            and self.agents_positions[j] == self.agents_positions[idx]
                        ):
                            # collision with other agent
                            # print("Collision occurred!")
                            rewards[agent] = self.reward_scheme.drones_collision
                            rewards[other_agent] = self.reward_scheme.drones_collision

                            if not collision_this_step:
                                self.num_collision_events += 1
                                collision_this_step = True

                            # Terminate episode for all agents (crash ends mission)
                            for a in self.agents:
                                terminations[a] = True
                                truncations[a] = True
                            break

                # No search if we moved (or left grid)
                continue

            # ------------------ SEARCH / DETECTION LOGIC ------------------
            # Only reach here if action == SEARCH
            cell_prob = float(prob_matrix[drone_y, drone_x])
            drone_found_person = False
            found_human = None

            for human in self.persons_set:
                if human.x == drone_x and human.y == drone_y:
                    drone_found_person = True
                    found_human = human
                    break

            # Detection attempt if a person is in this cell
            detection_success = False
            if drone_found_person and found_human is not None:
                random_value = random()
                max_detection_probability = min(
                    found_human.get_mult() * self.drone.pod, 1.0
                )
                detection_success = random_value <= max_detection_probability

            if detection_success:
                # Successful detection of the shipwrecked person
                if self.time_to_first_detection is None:
                    self.time_to_first_detection = self.timestep

                self.persons_set.remove(found_human)

                # Reward: 10000 + 10000 * (1 - timestep / timestep_limit)
                time_factor = 1.0 - (self.timestep / self.timestep_limit)
                base_reward = self.reward_scheme.search_and_find + (
                    self.reward_scheme.search_and_find * time_factor
                )

                if self.use_global_reward:
                    # Team reward
                    for a in self.agents:
                        rewards[a] = base_reward
                else:
                    rewards[agent] = base_reward

                if len(self.persons_set) == 0:
                    self.time_to_last_detection = self.timestep

                # If no persons remain, end the episode for all
                if len(self.persons_set) == 0:
                    person_found = True
                    for a in self.agents:
                        terminations[a] = True
                        truncations[a] = True

            else:
                # Search did NOT successfully detect a person in this cell
                # (either none present or detection failed). Reward depends
                # on the probability of the cell in the probability map.
                if cell_prob < 0.01:
                    # Cell prob < 1% -> -100
                    search_reward = self.reward_scheme.search_cell
                else:
                    # Cell prob >= 1% -> prob * 10000
                    search_reward = cell_prob * 10000.0

                if self.use_global_reward:
                    for a in self.agents:
                        rewards[a] = search_reward
                else:
                    rewards[agent] = search_reward

        # ------------------ POST-STEP BOOKKEEPING ------------------
        self.timestep += 1

        # Accumulate rewards for this step
        for a in self.agents:
            self.rewards_sum[a] += rewards[a]

        # Build infos with episode-level metrics
        infos = {
            drone: {
                "Found": person_found,
                "time_to_first_detection": self.time_to_first_detection,
                "time_to_last_detection": self.time_to_last_detection,
                "num_leave_grid_events": self.num_leave_grid_events,
                "num_collision_events": self.num_collision_events,
            }
            for drone in self.agents
        }

        self.render_step(any(terminations.values()), person_found)

        # Get observations for next step (this also advances prob matrix & persons)
        observations = self.create_observations()

        # If terminated, reset the agents list (PettingZoo parallel env requirement)
        if any(terminations.values()) or any(truncations.values()):
            self.agents = []

        return observations, rewards, terminations, truncations, infos

    def render_step(self, terminal, person_found):
        if self.render_mode == "human":
            if terminal:
                if person_found:
                    self.pygame_renderer.render_episode_end_screen(
                        f"The target was found in {self.timestep} moves", GREEN
                    )
                else:
                    self.pygame_renderer.render_episode_end_screen(
                        "The target was not found.", RED
                    )
            else:
                self.render()

    def build_movement_matrix(self, person: Person) -> np.ndarray:
        """
        Builds and outputs a 3x3 matrix from the probability matrix to use in the person movement function.
        """
        # Boundaries for the 3x3 movement matrix.
        left_x = max(person.x - 1, 0)
        right_x = min(person.x + 2, self.grid_size)
        left_y = max(person.y - 1, 0)
        right_y = min(person.y + 2, self.grid_size)

        probability_matrix = self.probability_matrix.get_matrix()
        movement_map = probability_matrix[left_y:right_y, left_x:right_x].copy()

        # Pad the matrix
        if person.x == 0:
            movement_map = np.insert(movement_map, 0, 0, axis=1)
        elif person.x == self.grid_size - 1:
            movement_map = np.insert(movement_map, 2, 0, axis=1)

        if person.y == 0:
            movement_map = np.insert(movement_map, 0, 0, axis=0)
        elif person.y == self.grid_size - 1:
            movement_map = np.insert(movement_map, 2, 0, axis=0)

        return movement_map

    def get_persons(self):
        return self.persons_set

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # Observation space for each agent:
        # - MultiDiscrete: (x, y) position of the agent
        # - Box: Probability matrix
        return Tuple(
            (
                MultiDiscrete([self.grid_size, self.grid_size]),
                Box(
                    low=0,
                    high=1,
                    shape=(self.grid_size, self.grid_size),
                    dtype=np.float32,
                ),
            )
        )

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(len(self.possible_actions))
