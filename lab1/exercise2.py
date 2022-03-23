import sys

import gym
import numpy as np

sys.path.append('av2')
from mdp import value_iteration


def simulate(env, iterations, discount_factor):
    steps_list = []
    reward_average_list = []

    for _ in range(iterations):
        env.reset()
        policy, value = value_iteration(env, discount_factor=discount_factor)
        # print(policy)
        # print(value)

        state = env.reset()
        # env.render()
        done = False
        steps = 0
        reward_sum = 0
        while not done:
            new_action = np.argmax(policy[state])
            state, reward, done, _ = env.step(new_action)
            reward_sum += reward
            # print(reward)
            # env.render()
            # time.sleep(0.5)
            steps += 1
            # os.system('cls')
        steps_list.append(steps)
        reward_average_list.append(reward_sum / steps)

    print(
        f'Discount factor: {discount_factor}'
        f', Average steps: {sum(steps_list) / iterations}'
        f', Average reward: {sum(reward_average_list) / iterations}'
    )


def main():
    env = gym.make('Taxi-v3')

    DISCOUNT_FACTORS = (0.5, 0.7, 0.9)
    ITERATIONS = (50, 100)

    for iterations in ITERATIONS:
        print(f'Iterations: {iterations}')
        for discount_factor in DISCOUNT_FACTORS:
            simulate(env, iterations, discount_factor)


if __name__ == '__main__':
    main()
