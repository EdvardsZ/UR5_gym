import functools
import time
import os
import gym
from stable_baselines3 import TD3, DDPG, SAC
from sb3_contrib import TQC
import gym
from gym_ur5 import randomizers

env_id = "PickAndPlaceUR5-v0"
total_timesteps = 250000
algorithm = TQC

algorithm_name = algorithm.__name__


def make_env_from_id(env_id: str, **kwargs) -> gym.Env:
    import gym
    return gym.make(env_id, **kwargs)


make_env = functools.partial(make_env_from_id, env_id=env_id)

env = randomizers.ur5_rg2_pick_and_place_no_rand.PickAndPLaceEnvNoRandomizations(env=make_env)

# Enable the rendering
env.render('human')

# Initialize the seed(for rng)
env.seed(42)
observation = env.reset()

algorithm_path = os.path.join('Training', 'Saved Models',algorithm_name+'_UR5PickAndPlaceIgnition'+str(total_timesteps) + env_id)
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

env.close()
time.sleep(5)