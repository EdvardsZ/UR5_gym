env_name = 'FetchReach-v1'
import os
import gym
from stable_baselines3 import HerReplayBuffer, DDPG, DQN, SAC, TD3
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
from stable_baselines3.common.vec_env import DummyVecEnv
from sb3_contrib import TQC

algorithm = TQC
algorithm_name = algorithm.__name__
timesteps = 50000

log_path = os.path.join('Training', 'Logs');
env = gym.make(env_name)
env = DummyVecEnv([lambda: env])

model = algorithm(
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
counter = 0
step = 50000
while counter < timesteps:
    counter += step
    model.learn(total_timesteps=step, reset_num_timesteps=False)
    algorithm_path = os.path.join('Training', 'Saved Models', algorithm_name+'_FetchPickAndPlaceMujoco_'+str(counter))
    model.save(algorithm_path)