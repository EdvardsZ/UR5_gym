import functools
import time

import gym
from gym_ignition.utils import logger
#from gym_ignition_environments import randomizers
from gym_ur5 import randomizers
from gym_ur5.tasks import reach
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
env.render()

# Initialize the seed
env.seed(42)
def solveReach(observation):
    end_effector = observation[:3]
    target = observation[3:]

    action = target-end_effector

    for i in range(3):
        action[i] = max(-1,action[i])
        action[i] = min(1,action[i])
    return action

for epoch in range(9):

    # Reset the environment
    observation = env.reset()

    # Initialize returned values
    done = False
    totalReward = 0

    while not done:

        action = solveReach(observation)

        observation, reward, done, _ = env.step(action)
        if(True):
            print('------')
            print("Observation:", observation)
            print("Reward:", reward)
            print("Done:", done)
            print("Info:", _)

        # It is not required to call this in the loop if physics is not randomized.
        env.render()
        # Accumulate the reward
        totalReward += reward
    #time.sleep(5)
    print(f"Reward episode #{epoch}: {totalReward}")

env.close()
time.sleep(5)