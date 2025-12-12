# simple_mcts.py
from __future__ import annotations

import copy
import itertools
import math
import random
from typing import Dict, List, Optional, Sequence, Tuple

# Exploration constant for UCB
UCB_C = 1.4


class MCTSNode:
    """
    Tree node for centralized multi-drone MCTS.

    - Each node corresponds to a *state* reachable from the root.
    - Edges are labeled with a *joint action* (tuple of ints).
    """

    def __init__(
        self,
        parent: Optional["MCTSNode"],
        joint_action_from_parent: Optional[Tuple[int, ...]],
        all_joint_actions: Sequence[Tuple[int, ...]],
    ):
        self.parent: Optional[MCTSNode] = parent
        # joint action that led from parent -> this node
        self.joint_action_from_parent: Optional[Tuple[int, ...]] = (
            joint_action_from_parent
        )

        self.children: Dict[Tuple[int, ...], MCTSNode] = {}
        self.visits: int = 0
        self.total_value: float = 0.0

        # Copy so each node tracks its own untried set
        self.untried_joint_actions: List[Tuple[int, ...]] = list(all_joint_actions)

    # --- helpers ---

    def q(self) -> float:
        """Mean value estimate."""
        return self.total_value / self.visits if self.visits > 0 else 0.0

    def is_fully_expanded(self) -> bool:
        return len(self.untried_joint_actions) == 0

    def best_child(self, c: float = UCB_C) -> "MCTSNode":
        """UCB1 selection among children."""
        assert self.children, "No children to select from"
        log_n = math.log(self.visits + 1e-8)

        def ucb_score(child: MCTSNode) -> float:
            return child.q() + c * math.sqrt(log_n / (child.visits + 1e-8))

        return max(self.children.values(), key=ucb_score)


def _enumerate_joint_actions(
    env,
    agents: Sequence[str],
) -> List[Tuple[int, ...]]:
    """
    Enumerate all joint actions for a set of agents, assuming each agent has the
    same discrete action space.

    joint_action = (a_0, a_1, ..., a_{N-1})
    """
    if not agents:
        return []

    # Assume same Discrete action space for all drones
    example_space = env.action_space(agents[0])
    n_actions = example_space.n
    single_actions = list(range(n_actions))

    # Cartesian product over all agents
    joint_actions = list(itertools.product(single_actions, repeat=len(agents)))
    return joint_actions


def mcts_plan_centralized(
    env,
    agents: Sequence[str],
    num_simulations: int = 50,
    max_rollout_depth: int = 100,
) -> Dict[str, int]:
    """
    Centralized multi-drone MCTS planner.

    Parameters
    ----------
    env : PettingZoo parallel env (e.g., DroneSwarmSearch)
        The *live* environment, at the current timestep/state.
    agents : list[str]
        Current list of drone agent ids (env.agents or env.get_agents()).
        The order is used consistently to define the joint-action tuple.
    num_simulations : int
        Number of MCTS simulations (tree traversals) to run from the current state.
    max_rollout_depth : int
        Max additional steps in a rollout beyond the current tree depth.

    Returns
    -------
    actions_dict : dict[str, int]
        A joint action dict mapping each agent_id -> action_index
        that you can pass directly to env.step(actions_dict).
    """
    if not agents:
        return {}

    # Fix a stable order of agents for this planning call
    agent_order: List[str] = list(agents)

    # Deep-copy the environment to use as root simulation env
    root_env = copy.deepcopy(env)

    # Build full joint-action space for this set of agents
    all_joint_actions = _enumerate_joint_actions(root_env, agent_order)
    if not all_joint_actions:
        # Degenerate case: no actions known; do nothing
        return {}

    # Root node: no action from parent (None), with full joint-action set
    root = MCTSNode(
        parent=None,
        joint_action_from_parent=None,
        all_joint_actions=all_joint_actions,
    )

    # --- MCTS main loop ---
    for _ in range(num_simulations):
        # 1. SELECTION
        node = root
        sim_env = copy.deepcopy(root_env)  # fresh clone per simulation
        done = False
        depth = 0
        sim_return = 0.0  # accumulate mean reward across drones

        # Walk down the tree while fully expanded and non-terminal
        while not done and node.is_fully_expanded() and node.children:
            node = node.best_child()

            ja = node.joint_action_from_parent
            assert ja is not None, "Child node must have a joint action from parent"

            actions_dict = {agent_id: int(a) for agent_id, a in zip(agent_order, ja)}

            obs, rewards, terminations, truncations, infos = sim_env.step(actions_dict)

            # Aggregate reward across drones (mean)
            if rewards:
                step_reward = sum(float(r) for r in rewards.values()) / len(agent_order)
                sim_return += step_reward

            done = any(terminations.values()) or any(truncations.values())
            depth += 1

        # 2. EXPANSION
        if not done and node.untried_joint_actions:
            ja = random.choice(node.untried_joint_actions)
            node.untried_joint_actions.remove(ja)

            actions_dict = {agent_id: int(a) for agent_id, a in zip(agent_order, ja)}

            obs, rewards, terminations, truncations, infos = sim_env.step(actions_dict)

            if rewards:
                step_reward = sum(float(r) for r in rewards.values()) / len(agent_order)
                sim_return += step_reward

            done = any(terminations.values()) or any(truncations.values())
            depth += 1

            child = MCTSNode(
                parent=node,
                joint_action_from_parent=ja,
                all_joint_actions=all_joint_actions,
            )
            node.children[ja] = child
            node = child

        # 3. ROLLOUT
        rollout_depth = 0
        while not done and rollout_depth < max_rollout_depth:
            # Random joint action
            ja = random.choice(all_joint_actions)
            actions_dict = {agent_id: int(a) for agent_id, a in zip(agent_order, ja)}

            obs, rewards, terminations, truncations, infos = sim_env.step(actions_dict)

            if rewards:
                step_reward = sum(float(r) for r in rewards.values()) / len(agent_order)
                sim_return += step_reward

            done = any(terminations.values()) or any(truncations.values())
            rollout_depth += 1
            depth += 1

        # 4. BACKPROPAGATION
        value = sim_return
        while node is not None:
            node.visits += 1
            node.total_value += value
            node = node.parent

    # --- Final action selection at root ---
    if not root.children:
        # If nothing was expanded (weird, but just in case), pick random joint action
        ja = random.choice(all_joint_actions)
    else:
        # Choose the child with the highest mean value
        best_ja, best_child = max(
            root.children.items(),
            key=lambda kv: kv[1].q(),
        )
        ja = best_ja

    # Convert joint action tuple back into the dict expected by env.step
    chosen_actions = {agent_id: int(a) for agent_id, a in zip(agent_order, ja)}
    return chosen_actions
