import gym
import gym_maze

if __name__ == '__main__':
    # env = gym.make('maze-sample-10x10-v0')
    # env = gym.make('maze-random-10x10-v0')
    env = gym.make('maze-random-10x10-plus-v0')

    env.reset()

    env.render()

    env.step('E')

    env.render()

    print()