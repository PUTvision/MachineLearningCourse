import os
import random
import time

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt


# use following command to install required package and all the dependencies:
# pip install gym[box2d,atari]
# for windows replace one of the atari files:
# pip install -f https://github.com/Kojoley/atari-py/releases atari_py


def ex_00():
    env = gym.make('CartPole-v1', render_mode='human')  # utworzenie środowiska
    env.reset()  # reset środowiska do stanu początkowego
    for _ in range(1000):  # kolejne kroki symulacji
        env.render()  # renderowanie obrazu
        action = env.action_space.sample()  # wybór akcji (tutaj: losowa akcja)
        env.step(action)  # wykonanie akcji
    env.close()  # zamknięcie środowiska


def ex_01():
    env = gym.make('MountainCar-v0', render_mode='human')

    # Observation and action space
    obs_space = env.observation_space
    action_space = env.action_space
    print("The observation space: {}".format(obs_space))
    print("The action space: {}".format(action_space))

    def policy(observation):
        return env.action_space.sample()

    # env = gym.make("LunarLander-v3")
    observation, info = env.reset(seed=42)
    for _ in range(10):
        env.render()

        action = policy(observation)  # User-defined policy function
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            observation, info = env.reset()
    env.close()

    env = gym.make('CartPole-v1', render_mode='human')  # utworzenie środowiska
    env.reset(seed=42)  # reset środowiska do stanu początkowego

    print(f'{env.action_space=}')
    print(f'{env.observation_space=}')
    print(f'{env.observation_space.high=}')
    print(f'{env.observation_space.low=}')

    action = 0
    env.reset()  # reset środowiska do stanu początkowego
    for _ in range(1000):  # kolejne kroki symulacji
        env.render()  # renderowanie obrazu
        time.sleep(0.05)

        # completely random action
        action = env.action_space.sample()  # wybór akcji (tutaj: losowa akcja)

        # flip-flopping the action
        # if action:
        #     action = 0
        # else:
        #     action = 1

        observation, reward, terminated, truncated, info = env.step(action)  # wykonanie akcji
        print(f'{action=}, {observation=}, {reward=}, {info=}')
        if terminated or truncated:
            print('The end!')
            time.sleep(1)
            env.reset()
    env.close()  # zamknięcie środowiska


def plot_outcoms(outcomes):
    plt.figure(figsize=(12, 5))
    plt.xlabel("Run number")
    plt.ylabel("Outcome")
    ax = plt.gca()
    ax.set_facecolor('#efeeea')
    plt.bar(range(len(outcomes)), outcomes, color="#0A047A", width=1.0)
    plt.show()


def ex_02():
    is_slippery = True

    env = gym.make('FrozenLake-v1', is_slippery=is_slippery)  # utworzenie środowiska
    env.reset(seed=42)

    print(f'{env.action_space=}')
    print(f'{env.observation_space=}')

    q_table = np.zeros((env.observation_space.n, env.action_space.n), dtype=np.float64)

    lr = 0.8 # 0.5
    discount_factor = 0.95 # 0.90
    # epsilon = 0.5
    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay_rate = 0.001

    no_training_episodes = 50000
    train_reward = 0
    outcomes = []
    current_epsilon = epsilon_start

    for i in range(no_training_episodes):  # kolejne kroki symulacji
        observation, info = env.reset()  # reset środowiska do stanu początkowego
        done = False
        total_reward = 0
        outcomes.append("Failure")
        while not done:
            if random.uniform(0, 1) < current_epsilon:
                # eksploracja
                action = env.action_space.sample()  # wybór akcji (tutaj: losowa akcja)
            else:
                # eksploatacja
                action = np.argmax(q_table[observation])
            next_observation, reward, terminated, truncated, info = env.step(action)  # wykonanie akcji
            done = terminated or truncated
            total_reward += reward

            # print(f'{action=}, {observation=}, {reward=}, {info=}')

            max_next_observation = np.max(q_table[next_observation])

            q_table[observation, action] = \
                ((1.0 - lr) * q_table[observation, action] +
                 lr * (reward + discount_factor * max_next_observation))

            observation = next_observation

            current_epsilon = max(epsilon_end, current_epsilon * (1 - epsilon_decay_rate))

        train_reward += total_reward
        if total_reward:
            outcomes[-1] = "Success"

    print(f'train_reward: {train_reward}')

    # plot_outcoms(outcomes)

    no_test_episodes = 100
    test_reward = 0

    print(q_table)

    env = gym.make('FrozenLake-v1', is_slippery=is_slippery, render_mode='human')

    for i in range(no_test_episodes):  # kolejne kroki symulacji
        observation, info = env.reset()  # reset środowiska do stanu początkowego
        env.render()
        done = False
        total_reward = 0
        while not done:
            action = np.argmax(q_table[observation])
            observation, reward, terminated, truncated, info = env.step(action)  # wykonanie akcji
            total_reward += reward
            done = terminated or truncated
            # print(f'observation: {observation}, reward: {reward}, info: {info}')
            env.render()  # renderowanie obrazu
        test_reward += total_reward
    print(f'test_reward: {test_reward}')

    env.close()  # zamknięcie środowiska


def main():
    # ex_00()
    # ex_01()
    ex_02()


if __name__ == '__main__':
    main()
