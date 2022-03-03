import gym

if __name__ == '__main__':
    env = gym.make('FrozenLake-v0')

    env.reset()

    env.render()

    state, reward, done, info = env.step(2)

    action = env.action_space.sample()

    state, reward, done, info = env.step(action)

    env.render()
