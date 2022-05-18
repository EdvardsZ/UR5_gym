import numpy
from gym.envs.registration import register
import gym_ignition.runtimes.gazebo_runtime
from . import tasks, models, randomizers

max_float = float(numpy.finfo(numpy.float32).max)

register(
    id="ReachUR5-v0",
    entry_point="gym_ignition.runtimes.gazebo_runtime:GazeboRuntime",
    max_episode_steps=100,
    kwargs={
        "task_cls": tasks.reach.Reach,
        "agent_rate": 2.5,
        "physics_rate": 100.0,
        "real_time_factor": max_float,
    },
)

register(
    id="ReachDictUR5-v0",
    entry_point="gym_ignition.runtimes.gazebo_runtime:GazeboRuntime",
    max_episode_steps=50,
    kwargs={
        "task_cls": tasks.reach_dict.Reach,
        "agent_rate": 2.5,
        "physics_rate": 100.0,
        "real_time_factor": max_float,
    },
)



register(
    id="PickAndPlaceUR5-v0",
    entry_point="gym_ignition.runtimes.gazebo_runtime:GazeboRuntime",
    max_episode_steps=100,
    kwargs={
        "task_cls": tasks.pick_and_place.PickAndPlace,
        "agent_rate": 2.5,
        "physics_rate": 500.0,
        "real_time_factor": 10,
    },
)