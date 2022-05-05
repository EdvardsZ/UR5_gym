env_name = 'FetchReach-v1'
import gym
import numpy as np
env = gym.make(env_name)

def solveReach(observation):
    end_effector = observation["achieved_goal"][:3]
    target = observation["desired_goal"][:3]

    action = (target-end_effector) * 20

    for i in range(3):
        action[i] = max(-1,action[i])
        action[i] = min(1,action[i])

    print(action)
    action = np.append(action, [0.0])
    print(action)
    return action


episodes = 5
for episode in range(1, episodes + 1):
    n_state = env.reset()
    done = False
    score = 0

    while not done:
        env.render()
        action = env.action_space.sample()
        #print("Action", action)
        action = solveReach(n_state)
        n_state, reward, done, info = env.step(action)
        #print("Observartion", n_state, reward, done, info)
        print('desired',n_state["desired_goal"])
        print('achived',n_state["achieved_goal"])
        score += reward
    print('Episonde:{} Score:{}'.format(episode, score))
env.close()