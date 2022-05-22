import itertools
import sys
import time
from typing import List, Tuple

import gym
import pandas as pd
import tqdm
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from deep_q_learning import DDQN

Parameters = Tuple[Tuple[float, float, int, float, int]]


def build_model(state_space_shape, num_actions, learning_rate):
    model = Sequential()

    model.add(Dense(32, activation='relu', input_dim=state_space_shape))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(num_actions, activation='linear'))

    model.compile(Adam(lr=learning_rate), loss=MeanSquaredError())

    return model


def fit_agent(env: gym.Wrapper, episodes: int, decay: float, agent: DDQN,
              epsilon_min: float, epsilon: float):
    """Fits the agent to the given environment.

    :param env: OpenAI Gym environment
    :param episodes: number of episodes to train the agent
    :param decay: decay rate for the epsilon
    :param agent: agent to fit
    :param epsilon_min: minimum epsilon
    :param epsilon: current epsilon
    """
    for _ in tqdm.tqdm(range(episodes), desc=f'Training, episodes={episodes}'):
        state = env.reset()
        done = False
        while not done:
            action = agent.get_action(state, epsilon)
            new_state, reward, done, _ = env.step(action)
            agent.update_memory(state, action, reward,
                                new_state, done)
            state = new_state
        agent.train()
        agent.update_target_model()
        if epsilon > epsilon_min:
            epsilon -= decay


def evaluate_agent(env: gym.Wrapper, agent: DDQN, num_episodes: int
                   ) -> Tuple[float, float]:
    """
    Evaluates the given q table on the given environment.
    :param env: OpenAI Gym environment
    :param agent: agent to evaluate
    :param num_episodes: number of episodes to evaluate the policy
    :return: average steps and average rewards for the given policy
    """
    total_steps = 0
    total_reward = 0.0
    for _ in tqdm.tqdm(
            range(num_episodes), desc=f'Evaluating, iterations={num_episodes}'
    ):
        state = env.reset()
        done = False
        steps = 0
        reward_sum = 0
        while not done:
            action = agent.get_action(state, epsilon=0)
            state, reward, done, _ = env.step(action)
            reward_sum += reward
            steps += 1
        total_steps += steps
        total_reward += reward_sum

    average_steps = total_steps / num_episodes
    average_reward = total_reward / num_episodes
    return average_steps, average_reward


def run_experiments(env: gym.Wrapper, parameters: Parameters, visualize: bool
                    ) -> List[dict]:
    """
    Runs the experiments for the given parameters.
    :param env: OpenAI Gym environment
    :param parameters: parameters to run the experiments
    :param visualize: whether to visualize the results
    :return: list of results
    """
    results_list = []

    for (
        i, (discount_factor, learning_rate, episodes, decay, iterations)
    ) in enumerate(parameters):
        print(
            f'[{i+1}/{len(parameters)}]',
            f'{discount_factor=}, {learning_rate=}, {episodes=}, {decay=}'
            f', {iterations=}'
        )
        num_actions = env.action_space.n
        state_space_shape = env.observation_space.shape[0]
        model = build_model(state_space_shape, num_actions, learning_rate)
        target_model = build_model(state_space_shape, num_actions,
                                   learning_rate)
        agent = DDQN(
            state_space_shape, num_actions, model, target_model,
            learning_rate, discount_factor, batch_size=128, memory_size=1024
        )

        epsilon_min = 0.1
        if decay == 0:
            epsilon = epsilon_min
        else:
            epsilon = 0.25

        fit_agent(env, episodes, decay, agent, epsilon_min, epsilon)

        average_steps, average_reward = evaluate_agent(
            env, agent, iterations
        )

        results = {
            'discount_factor': discount_factor,
            'learning_rate': learning_rate,
            'episodes': episodes,
            'decay': decay,
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
                new_action = agent.get_action(state, epsilon=0)
                state, reward, done, _ = env.step(new_action)
                # print(reward)
                env.render()
                time.sleep(0.5)

        print('-=' * 40)

    return results_list


def main():
    env = gym.make('MountainCar-v0')

    visualize = len(sys.argv) > 1 and sys.argv[1] in ('-v', '--visualize')
    # Kratam na kombinacii od prethodnata lab za da moze za pokratko da zavrshi
    DISCOUNT_FACTORS = (0.9,)
    LEARNING_RATES = (0.01,)
    EPISODES = (5, 10, 15)
    DECAYS = (0.01,)
    ITERATIONS = (50, 100)

    parameters = tuple(itertools.product(
        DISCOUNT_FACTORS, LEARNING_RATES, EPISODES, DECAYS, ITERATIONS
    ))

    results_list = run_experiments(env, parameters, visualize)

    df_results = pd.DataFrame(results_list)
    df_results.to_csv('exercise2_results.csv', index=False)


if __name__ == '__main__':
    main()
