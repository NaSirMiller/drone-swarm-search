# Project Overview

This codebase is a fork of the [Drone Swarm Search Environment](https://github.com/pfeinsper/drone-swarm-search-algorithms) and was adapted, where needed to
support learning methods.

## Instructions to Run

### Single Simulation with Specific Policy

The `test.py` program outlines a single simulation for agents in the DSSE with a specific policy. To change the policy, you may need to manually import from other parts of the repository.

```
python test.py
```

### Evaluation across Policies

The `evaluation.py` program runs a specified number of simulations for a selected policy and outputs the agent's performance in a set of metrics.

```
python evaluation.py
```

However, if you want to run the exact evaluation pipeline I did for the corresponding paper, run the following script:

```
./run_all_simulations.sh
```
