from typing import Tuple

import gym
import numpy as np
import tqdm

from deep_q_learning import DDPG, OrnsteinUhlenbeckActionNoise


def preprocess_reward(reward):
    new_reward = np.clip(reward, -5., 5.)
    return new_reward


def evaluate_agent(env: gym.Wrapper, agent: DDPG, num_episodes: int
                   ) -> Tuple[float, float]:
    """Evaluates the given agent on the given environment.

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
        reward_sum = 0.0
        while not done:
            action = agent.get_action(state, discrete=False)
            state, reward, done, _ = env.step(action)
            reward = preprocess_reward(reward)
            reward_sum += reward
            steps += 1
        total_steps += steps
        total_reward += reward_sum

    average_steps = total_steps / num_episodes
    average_reward = total_reward / num_episodes
    return average_steps, average_reward


def main():
    env = gym.make('LunarLander-v2', continuous=True)
    env.reset()
    # env.render()

    state_space_shape = env.observation_space.shape[0]
    action_space_shape = env.action_space.shape[0]

    num_episodes = 30
    learning_rate_actor = 0.01
    learning_rate_critic = 0.02
    discount_factor = 0.99
    batch_size = 64
    memory_size = 1000

    noise = OrnsteinUhlenbeckActionNoise(action_space_shape)

    agent = DDPG(
        state_space_shape, action_space_shape, learning_rate_actor,
        learning_rate_critic, discount_factor, batch_size, memory_size
    )

    agent.build_model()

    for episode in range(num_episodes):
        state = env.reset()
        # env.render()
        done = False
        while not done:
            action = agent.get_action(state, discrete=False) + noise()
            next_state, reward, done, _ = env.step(action)
            # env.render()
            reward = preprocess_reward(reward)
            numeric_done = 1 if done else 0
            agent.update_memory(
                state, action, reward, next_state, numeric_done
            )
            state = next_state
        agent.train()
        if episode % 5 == 0:
            agent.update_target_model()
        if episode % 50 == 0:
            agent.save('LunarLander-v2', episode)

    for iterations in (50, 100):
        average_steps, average_reward = evaluate_agent(
            env, agent, iterations
        )
        print(
            f'Iterations: {iterations}, Average steps: {average_steps}, '
            f'Average reward: {average_reward}'
        )


if __name__ == '__main__':
    main()
