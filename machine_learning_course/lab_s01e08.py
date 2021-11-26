import gym
import numpy as np

import time
import random
import os


def ex_01():
    env = gym.make('CartPole-v1')  # utworzenie środowiska
    env.seed(42)

    print(env.action_space)
    print(env.observation_space)
    print(env.observation_space.high)
    print(env.observation_space.low)

    action = 1
    env.reset()  # reset środowiska do stanu początkowego
    for _ in range(1000):  # kolejne kroki symulacji
        env.render()  # renderowanie obrazu
        time.sleep(0.1)
        if action:
            # action = env.action_space.sample()  # wybór akcji (tutaj: losowa akcja)
            action = 0
        else:
            action = 1
        print(action)
        observation, reward, done, info = env.step(action)  # wykonanie akcji
        print(f'observation: {observation}, reward: {reward}, info: {info}')
        if done:
            print('The end!')
            time.sleep(1)
    env.close()  # zamknięcie środowiska


def ex_02():
    env = gym.make('FrozenLake-v0')  # utworzenie środowiska
    env.seed(42)

    print(env.action_space)
    print(env.observation_space)

    q_table = np.zeros((env.observation_space.n, env.action_space.n), dtype=np.float64)
    lr = 0.1
    discount_factor = 0.6
    epsilon = 0.5

    no_training_episodes = 10000

    for i in range(no_training_episodes):  # kolejne kroki symulacji
        observation = env.reset()  # reset środowiska do stanu początkowego
        # env.render()
        done = False
        total_reward = 0
        while not done:
            if random.uniform(0, 1) < epsilon:
                # eksploracja
                action = env.action_space.sample()  # wybór akcji (tutaj: losowa akcja)
            else:
                # eksploatacja
                action = np.argmax(q_table[observation])
            next_observation, reward, done, info = env.step(action)  # wykonanie akcji
            total_reward += reward
            # print(f'observation: {observation}, reward: {reward}, info: {info}')

            max_next_observation = np.max(q_table[next_observation])

            q_table[observation, action] = (1 - lr) * q_table[observation, action] + lr * (reward + discount_factor * max_next_observation)

            observation = next_observation

            # env.render()  # renderowanie obrazu
            # time.sleep(0.1)
        if i % 100 == 0:
            print(f'total_reward episode: {i+1}: {total_reward}')

    no_test_episodes = 100

    for i in range(no_test_episodes):  # kolejne kroki symulacji
        observation = env.reset()  # reset środowiska do stanu początkowego
        env.render()
        done = False
        total_reward = 0
        while not done:
            action = np.argmax(q_table[observation])
            observation, reward, done, info = env.step(action)  # wykonanie akcji
            total_reward += reward
            print(f'observation: {observation}, reward: {reward}, info: {info}')
            os.system('cls')
            env.render()  # renderowanie obrazu
            time.sleep(0.1)
        # if i % 100:
        print(f'total_reward episode: {i+1}: {total_reward}')
        input()

    env.close()  # zamknięcie środowiska


def main():
    # ex_01()
    ex_02()


if __name__ == '__main__':
    main()
