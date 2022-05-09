import functools
import time
import os
import gym
from stable_baselines3 import PPO, DDPG, SAC
import gym
from gym_ur5 import randomizers
from stable_baselines3.common.evaluation import evaluate_policy

env_name = 'FetchPickAndPlace-v1'
total_timesteps = 600000
algorithm = SAC

algorithm_name = algorithm.__name__

env =  gym.make(env_name)

observation = env.reset()

algorithm_path = os.path.join('Training', 'Saved Models', 'SAC_FetchPickAndPlaceMujoco_600000')
model = algorithm.load(algorithm_path, env=env)

model = algorithm.load(algorithm_path, env=env)
for epoch in range(15):
    # Reset the environment
    observation = env.reset()

    # Initialize returned values
    done = False
    totalReward = 0

    while not done:

        # Execute a random action
        action = model.predict(observation)
        observation, reward, done, _ = env.step(action[0])
        env.render('human')
        totalReward += reward
    time.sleep(0)
    print(f"Reward episode #{epoch}: {totalReward}")