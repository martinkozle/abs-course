import gym
import numpy as np
from keras.layers import Activation, Concatenate, Dense, Flatten, Input
from keras.models import Model, Sequential
from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess
from tensorflow.keras.optimizers import Adam


def preprocess_reward(reward):
    new_reward = np.clip(reward, -5., 5.)
    return new_reward


def main():
    env = gym.make('LunarLander-v2', continuous=True)
    env.reset()

    nb_actions = env.action_space.shape[0]

    actor = Sequential()
    actor.add(Flatten(input_shape=(1,) + env.observation_space.shape))
    actor.add(Dense(64, activation='relu'))
    actor.add(Dense(64, activation='relu'))
    actor.add(Dense(nb_actions, activation='tanh'))
    print(actor.summary())

    action_input = Input(shape=(nb_actions,), name='action_input')
    observation_input = Input(
        shape=(1,) + env.observation_space.shape, name='observation_input'
    )
    flattened_observation = Flatten()(observation_input)
    x = Concatenate()([action_input, flattened_observation])
    x = Dense(32)(x)
    x = Activation('relu')(x)
    x = Dense(64)(x)
    x = Activation('relu')(x)
    x = Dense(32)(x)
    x = Activation('relu')(x)
    x = Dense(1)(x)
    x = Activation('linear')(x)
    critic = Model(inputs=[action_input, observation_input], outputs=x)
    print(critic.summary())

    memory = SequentialMemory(limit=100000, window_length=1)
    random_process = OrnsteinUhlenbeckProcess(
        size=nb_actions, theta=.15, mu=0., sigma=.3
    )
    agent = DDPGAgent(
        nb_actions=nb_actions, actor=actor, critic=critic,
        critic_action_input=action_input, memory=memory,
        nb_steps_warmup_critic=100, nb_steps_warmup_actor=100,
        random_process=random_process, gamma=.99, target_model_update=1e-3
    )
    agent.compile(Adam(lr=.001, clipnorm=1.), metrics=['mae'])

    agent.fit(env, nb_steps=50000, verbose=1, nb_max_episode_steps=200)
    agent.save_weights('ddpg_exercise2_weights.h5f', overwrite=True)

    agent.test(env, nb_episodes=50, nb_max_episode_steps=200)
    agent.test(env, nb_episodes=100, nb_max_episode_steps=200)


if __name__ == '__main__':
    main()
