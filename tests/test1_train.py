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

algorithm = SAC
total_timesteps = 50000

algorithm_name = algorithm.__name__

def make_env_from_id(env_id: str, **kwargs) -> gym.Env:
    import gym
    return gym.make(env_id, **kwargs)

make_env = functools.partial(make_env_from_id, env_id=env_name)

env = ur5_rg2_no_rand.ReachEnvNoRandomizations(env=make_env)
env.reset()
env.seed(42)

env = DummyVecEnv([lambda: env])

log_path = os.path.join('Training', 'Logs')
model = algorithm('MlpPolicy', env, verbose = 1, tensorboard_log = log_path)
model.learn(total_timesteps = total_timesteps)
algorithm_path = os.path.join('Training', 'Saved Models', algorithm_name+'_'+str(total_timesteps))
print("Saving model to:"+ algorithm_path)
model.save(algorithm_path)