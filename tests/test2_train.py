from stable_baselines3.her import GoalSelectionStrategy

env_id = "PickAndPlaceUR5-v0"
import functools
import time
import os
import gym
from stable_baselines3 import PPO, SAC, DDPG, HerReplayBuffer
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from sb3_contrib import TQC
from gym_ignition.utils import logger
from gym_ur5 import randomizers
from gym_ur5.randomizers import ur5_rg2_no_rand
from gym_ignition.runtimes import gazebo_runtime
from gym_ur5.tasks.reach_dict import Reach
algorithm = TQC
total_timesteps = 1000000
from gym.envs.robotics.fetch_env import FetchEnv
algorithm_name = algorithm.__name__

def make_env_from_id(env_id: str, **kwargs) -> gym.Env:
    import gym
    return gym.make(env_id, **kwargs)

make_env = functools.partial(make_env_from_id, env_id=env_id)

env = randomizers.ur5_rg2_pick_and_place_no_rand.PickAndPLaceEnvNoRandomizations(env=make_env)
env.reset()
env.seed(42)

env = DummyVecEnv([lambda: env])
log_path = os.path.join('Training', 'Logs')
model = algorithm(
    'MultiInputPolicy',
    env,
    verbose = 1,
    tensorboard_log = log_path,
    gamma= 0.95,
    learning_starts= 10000,
    replay_buffer_class = HerReplayBuffer,
    # Parameters for HER
    replay_buffer_kwargs=dict(
        n_sampled_goal = 4,
        goal_selection_strategy = GoalSelectionStrategy.FUTURE,
        online_sampling = False,
    ),
)
model.learn(total_timesteps = total_timesteps)
algorithm_path = os.path.join('Training', 'Saved Models', algorithm_name+'_'+str(total_timesteps) + env_id)
print("Saving model to:"+ algorithm_path)
model.save(algorithm_path)