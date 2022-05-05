env_name = 'FetchPickAndPlace-v1'
import gym

env = gym.make(env_name)

episodes = 5
for episode in range(1, episodes + 1):
    state = env.reset()
    done = False
    score = 0

    while not done:
        env.render()
        action = env.action_space.sample()
        print("Action", action)
        n_state, reward, done, info = env.step(action)
        print("Observartion", n_state, reward, done, info)
        score += reward
    print('Episonde:{} Score:{}'.format(episode, score))
env.close()