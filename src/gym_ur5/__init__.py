import numpy
from gym.envs.registration import register

from . import tasks, models, randomizers

max_float = float(numpy.finfo(numpy.float32).max)

register(
    id="Idiot-v1",
    entry_point="gym_ignition.runtimes.gazebo_runtime:GazeboRuntime",
    max_episode_steps=5000,
    kwargs={
        "task_cls": tasks.cartpole_discrete_balancing.CartPoleDiscreteBalancing,
        "agent_rate": 1000,
        "physics_rate": 1000,
        "real_time_factor": max_float,
    },
)
register(
    id="ReachUR5-v0",
    entry_point="gym_ignition.runtimes.gazebo_runtime:GazeboRuntime",
    max_episode_steps=10_000,
    kwargs={
        "task_cls": tasks.reach.Reach,
        "agent_rate": 1000,
        "physics_rate": 1000,
        "real_time_factor": max_float,
    },
)