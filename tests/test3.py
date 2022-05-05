import functools
import time
import gym

from gym_ur5 import randomizers

env_id = "ReachDictUR5-v0"


def make_env_from_id(env_id: str, **kwargs) -> gym.Env:
    import gym

    return gym.make(env_id, **kwargs)

make_env = functools.partial(make_env_from_id, env_id=env_id)

env = randomizers.ur5_rg2_no_rand.ReachEnvNoRandomizations(env=make_env)

env.render()

# Initialize the seed
env.seed(42)

for epoch in range(6):

    observation = env.reset()

    done = False
    totalReward = 0

    while not done:

        action = env.action_space.sample()
        observation, reward, done, _ = env.step(action)
        print('observation', observation['observation'])
        env.render()
        totalReward += reward
    #time.sleep(5)
    print(f"Reward episode #{epoch}: {totalReward}")

env.close()
time.sleep(5)