env_name = 'FetchReach-v1'
import os
import gym
from stable_baselines3 import HerReplayBuffer, DDPG, DQN, SAC, TD3
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
from stable_baselines3.common.vec_env import DummyVecEnv

log_path = os.path.join('Training', 'Logs');
env = gym.make(env_name)
env = DummyVecEnv([lambda: env])
model = SAC(
    'MultiInputPolicy',
    env,
    verbose = 1,
    tensorboard_log = log_path,
    replay_buffer_class = HerReplayBuffer,
    # Parameters for HER
    replay_buffer_kwargs=dict(
        n_sampled_goal = 4,
        goal_selection_strategy = GoalSelectionStrategy.FUTURE,
        online_sampling = False,
    ),
)
model.learn(total_timesteps = 100000)