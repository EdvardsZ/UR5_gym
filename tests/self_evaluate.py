import functools
import time
import os
import gym
from stable_baselines3 import PPO, DDPG, SAC
import gym
from gym_ur5 import randomizers

env_id = "ReachUR5-v0"
total_timesteps = 40000
algorithm = SAC

algorithm_name = algorithm.__name__


def make_env_from_id(env_id: str, **kwargs) -> gym.Env:
    import gym
    return gym.make(env_id, **kwargs)


make_env = functools.partial(make_env_from_id, env_id=env_id)

env = randomizers.ur5_rg2_no_rand.ReachEnvNoRandomizations(env=make_env)

# Enable the rendering
env.render('human')

# Initialize the seed(for rng)
env.seed(42)
observation = env.reset()

algorithm_path = os.path.join('Training', 'Saved Models', algorithm_name+'_'+str(total_timesteps))
model = algorithm.load(algorithm_path, env=env)
for epoch in range(10):
    # Reset the environment
    observation = env.reset()

    # Initialize returned values
    done = False
    totalReward = 0

    while not done:

        # Execute a random action
        action = env.action_space.sample()
        print('action:', action)
        action = model.predict(observation)
        print('action', action)
        print(action[0])
        observation, reward, done, _ = env.step(action[0])
        print(observation, reward, done, _)
        #input('test')
        # Render the environment.
        # It is not required to call this in the loop if physics is not randomized.
        env.render('human')
        # Accumulate the reward
        totalReward += reward
    time.sleep(5)
    print(f"Reward episode #{epoch}: {totalReward}")

env.close()
time.sleep(5)