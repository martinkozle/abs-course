import itertools
import os
import sys
import time
from typing import List, Tuple

import gym
import pandas as pd

from q_learning import (calculate_new_q_value, get_action, get_best_action,
                        random_q_table)


def fit_q_table(env, discount_factor, learning_rate, episodes, q_table,
                epsilon, epsilon_min, decay):
    """
    Fits the q table to the given environment.
    :param env: OpenAI Gym environment
    :param discount_factor: discount factor
    :param learning_rate: learning rate
    :param episodes: number of episodes to fit the policy
    :param q_table: n-dimensional q table
    :param epsilon: epsilon value
    :param epsilon_min: minimum epsilon value
    :param decay: decay value
    :return: None
    """
    for _ in range(episodes):
        state = env.reset()
        done = False
        while not done:
            action = get_action(env, q_table, state, epsilon)
            new_state, reward, done, _ = env.step(action)
            q_table[state, action] = calculate_new_q_value(
                q_table, state, new_state, action, reward, learning_rate,
                discount_factor
            )
            state = new_state
        if epsilon > epsilon_min:
            epsilon -= decay


def evaluate_q_table(env, q_table, num_episodes) -> Tuple[float, float]:
    """
    Evaluates the given q table on the given environment.
    :param env: OpenAI Gym environment
    :param q_table: n-dimensional q table
    :param num_episodes: number of episodes to evaluate the policy
    :return: average steps and average rewards for the given policy
    """
    total_steps = 0
    total_reward = 0.0
    for _ in range(num_episodes):
        state = env.reset()
        done = False
        steps = 0
        reward_sum = 0
        while not done:
            action = get_best_action(q_table, state)
            state, reward, done, _ = env.step(action)
            reward_sum += reward
            steps += 1
        total_steps += steps
        total_reward += reward_sum

    average_steps = total_steps / num_episodes
    average_reward = total_reward / num_episodes
    return average_steps, average_reward


def run_experiments(env, parameters, visualize) -> List[dict]:
    """
    Runs the experiments for the given parameters.
    :param env: OpenAI Gym environment
    :param parameters: parameters to run the experiments
    :param visualize: whether to visualize the results
    :return: list of results
    """
    results_list = []

    for (
        i, (discount_factor, learning_rate, episodes, iterations)
    ) in enumerate(parameters):
        print(
            f'[{i+1}/{len(parameters)}]',
            f'{discount_factor=}, {learning_rate=}, {episodes=}, {iterations=}'
        )
        num_actions = env.action_space.n
        num_states = env.observation_space.n
        q_table = random_q_table(-1, 0, (num_states, num_actions))
        epsilon = 0.25
        epsilon_min = 0.1
        decay = 0.05

        fit_q_table(env, discount_factor, learning_rate, episodes,
                    q_table, epsilon, epsilon_min, decay)

        print(q_table)

        average_steps, average_reward = evaluate_q_table(
            env, q_table, iterations
        )

        results = {
            'discount_factor': discount_factor,
            'learning_rate': learning_rate,
            'episodes': episodes,
            'iterations': iterations,
            'average_steps': average_steps,
            'average_reward': average_reward
        }

        print(results)

        results_list.append(results)

        if visualize:
            state = env.reset()
            env.render()
            steps = 0
            done = False
            while not done and steps < 20:
                steps += 1
                new_action = get_best_action(q_table, state)
                state, reward, done, _ = env.step(new_action)
                print(reward)
                env.render()
                time.sleep(0.5)
                os.system('cls')  # nosec

        print('-=' * 40)

    return results_list


def main():
    env = gym.make('Taxi-v3')

    visualize = len(sys.argv) > 1 and sys.argv[1] in ('-v', '--visualize')
    DISCOUNT_FACTORS = (0.5, 0.9)
    LEARNING_RATES = (0.1, 0.01)
    EPISODES = (10, 50, 100, 500, 1000, 5000, 10000)
    ITERATIONS = (50, 100)

    parameters = tuple(itertools.product(
        DISCOUNT_FACTORS, LEARNING_RATES, EPISODES, ITERATIONS
    ))

    results_list = run_experiments(env, parameters, visualize)

    df_results = pd.DataFrame(results_list)
    df_results.to_csv('exercise1_results.csv', index=False)


if __name__ == '__main__':
    main()
