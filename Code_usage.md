# Adapted Library
This repository just extended the behavorial cloning class to simulate the assistant model traning from the paper [ULD](https://arxiv.org/abs/2406.08607). The current implementation just works for discrete action spaces. Note that most function in ULD just work in a vectorized environment containing one environment


# Usage
```
from stable_baselines3 import DQN
from imitation.util import util
import imitation.data.rollout as rollout
from imitation.algorithms import bc
import gymnasium as gym
from imitation.util.util import make_vec_env
from imitation.data import serialize
import numpy as np

rng = np.random.default_rng(0)

venv = make_vec_env(
    "FrozenLake-v1",
    n_envs=4,
    rng=rng,
    env_make_kwargs={
        "map_name": "4x4",
        "is_slippery": True
    }
)

#load dataset
unlearn = list(serialize.load("alternate/trajectories/unlearn"))
retain = list(serialize.load("alternate/trajectories/retain"))


transitions_unlearn = rollout.flatten_trajectories(unlearn)
transitions_retain = rollout.flatten_trajectories(retain)

bc_trainer = bc.BC(
    observation_space=venv.observation_space,   
    action_space=venv.action_space,
    demonstrations=transitions_unlearn,
    demo_uni = transitions_retain,
    rng=rng,
    batch_size=35,
)
#train assistant
bc_trainer.assistant_train(n_transitions = 2800000, uni_rate  = 1)
```