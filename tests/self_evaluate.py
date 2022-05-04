import functools
import time
import os
import gym
from stable_baselines3 import PPO, DDPG
import gym
from gym_ignition.utils import logger
#from gym_ignition_environments import randomizers
from gym_ur5 import randomizers
# Set verbosity
logger.set_level(gym.logger.ERROR)
# logger.set_level(gym.logger.DEBUG)

# Available tasks
#env_id = "Pendulum-Gazebo-v0"
env_id = "ReachUR5-v0"
# env_id = "CartPoleContinuousBalancing-Gazebo-v0"
# env_id = "CartPoleContinuousSwingup-Gazebo-v0"


def make_env_from_id(env_id: str, **kwargs) -> gym.Env:
    import gym
    import gym_ignition_environments

    return gym.make(env_id, **kwargs)


# Create a partial function passing the environment id
make_env = functools.partial(make_env_from_id, env_id=env_id)

# Wrap the environment with the randomizer.
# This is a simple example no randomization are applied.
#envs = pendulum_no_rand.PendulumEnvNoRandomizations(envs=make_env)
env = randomizers.ur5_rg2_no_rand.ReachEnvNoRandomizations(env=make_env)
#env = randomizers.cartpole_no_rand.CartpoleEnvNoRandomizations(env=make_env)

# Wrap the environment with the randomizer.
# This is a complex example that randomizes both the physics and the model.
# envs = randomizers.cartpole.CartpoleEnvRandomizer(
#     envs=make_env, seed=42, num_physics_rollouts=5)

# Enable the rendering
env.render('human')

# Initialize the seed
env.seed(42)
observation = env.reset()
DDPG_Path = os.path.join('Training', 'Saved Models', 'DDPG_HER_50K')
model = DDPG.load(DDPG_Path, env=env)
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
    #time.sleep(5)
    print(f"Reward episode #{epoch}: {totalReward}")

env.close()
time.sleep(5)