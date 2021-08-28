import gym
import numpy as np
from dueling_ddqn_torch import Agent
from utils import plot_learning_curve

from gym import wrappers
import os


if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    n_games = 500
    load_checkpoint = False

    agent = Agent(gamma=0.99, epsilon=1.0, eps_min=0.01, eps_dec=1e-3,
                  lr=5e-4, input_dims=[8], n_actions=4, mem_size=1000000,
                  batch_size=64, replace=100)

    if load_checkpoint:
        agent.load_models()

    filename = 'LunarLander-DuelingDDQN-Adam-lr0005-replace100.png'
    scores, eps_history = [], []

    for i in range(n_games):
        done = False
        score = 0
        observation = env.reset()

        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            agent.store_transition(observation, action, reward, observation_, int(done))
            agent.learn()
            observation = observation_
            env.render()

        scores.append(score)
        avg_score = np.mean(scores[-100:])
        eps_history.append(agent.epsilon)
        print('episode ', i, ' | score %.2f' % score,
              ' | average score %.2f' % avg_score, ' | epsilon %.2f' % agent.epsilon)

        if i % 10 == 0 and i > 0:
            agent.save_models()
        if i == n_games-1:
            agent.save_models()

    x = [i+1 for i in range(n_games)]
    plot_learning_curve(x, scores, eps_history, filename)