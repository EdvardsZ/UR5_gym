import functools
import time
import gym
import numpy as np
from gym_ur5 import randomizers

env_id = "PickAndPlaceUR5-v0"


def make_env_from_id(env_id: str, **kwargs) -> gym.Env:
    import gym

    return gym.make(env_id, **kwargs)

make_env = functools.partial(make_env_from_id, env_id=env_id)

env = randomizers.ur5_rg2_pick_and_place_no_rand.PickAndPLaceEnvNoRandomizations(env=make_env)

env.render()

# Initialize the seed
env.seed(42)

def solveReach(observation):
    end_effector = observation["observation"][:3]
    target = observation["achieved_goal"][:3]

    if np.linalg.norm(target[:2]-end_effector[:2]) > 0.05:
        target[2] += 0.4

    action = (target-end_effector)

    for i in range(3):
        action[i] = max(-1, action[i])
        action[i] = min(1, action[i])

    action = np.append(action, [0.0])

    if np.linalg.norm(action) < 0.05:
        action[3] = -1.0

    return action

for epoch in range(10):

    observation = env.reset()

    done = False
    totalReward = 0

    while not done:

        action = env.action_space.sample()
        action = solveReach(observation)
        observation, reward, done, _ = env.step(action)
        #print('observation', observation)
        #print(reward)
        env.render()
        totalReward += reward
    #time.sleep(5)
    print(f"Reward episode #{epoch}: {totalReward}")

env.close()
time.sleep(5)