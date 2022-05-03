env_name = "ReachUR5-v0"
import functools
import time
import os
import gym
from stable_baselines3 import PPO, SAC, DDPG
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from gym_ignition.utils import logger
from gym_ignition_environments import randomizers
from gym_ur5.randomizers import ur5_rg2_no_rand

algorithm = DDPG
print(algorithm)
def make_env_from_id(env_id: str, **kwargs) -> gym.Env:
    import gym
    import gym_ignition_environments
    return gym.make(env_id, **kwargs)
    # Calling gym.make(env_id, **kwargs) uses file gym/envs/registration.py, in which is called function
    # make() from class EnvRegistry. Function make() uses as input arguments: id and kwargs.
    # Function make() returns created envs.
# Create a partial function passing the environment id
make_env = functools.partial(make_env_from_id, env_id=env_name)

# Wrap the environment with the randomizer.
# This is a simple example no randomization are applied.
env = ur5_rg2_no_rand.ReachEnvNoRandomizations(env=make_env)
env.reset()
env.seed(42)

env = DummyVecEnv([lambda: env])

log_path = os.path.join('Training', 'Logs');
env.render("human")
model = algorithm('MlpPolicy', env, verbose = 1, tensorboard_log = log_path)
model.learn(total_timesteps = 200_000)
algorithm_Path = os.path.join('Training', 'Saved Models', '+DDPGHERPendulum_swing_50k')
model.save(algorithm_Path)