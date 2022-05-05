import functools
import time

import gym
from gym_ignition.utils import logger
from gym_ur5 import randomizers

env_id = "ReachDictUR5-v0"


def make_env_from_id(env_id: str, **kwargs) -> gym.Env:
    import gym
    return gym.make(env_id, **kwargs)

# Create a partial function passing the environment id
make_env = functools.partial(make_env_from_id, env_id=env_id)
env = randomizers.ur5_rg2_no_rand.ReachEnvNoRandomizations(env=make_env)
env.render()
env.seed(42)
def solveReach(observation):
    end_effector = (observation["achieved_goal"])[:3]
    target = (observation["desired_goal"])[:3]

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

        env.render()
        # Accumulate the reward
        totalReward += reward
    #time.sleep(5)
    print(f"Reward episode #{epoch}: {totalReward}")
print(env.observation_space)
env.close()
time.sleep(5)