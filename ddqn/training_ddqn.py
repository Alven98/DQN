from ddqn import DDQNAgent
import numpy as np
from utils import plotLearning
import gym
from gym import wrappers
import os

if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    ddqn_agent = DDQNAgent(alpha=0.0005, gamma=0.99, n_actions=4, epsilon=1.0, input_dims=8,
                  mem_size=1000000, batch_size=64, epsilon_end=0.01)

    n_games = 500
    ddqn_agent.load_model()
    ddqn_scores = []
    eps_history = []

    # env = wrappers.Monitor(env, os.path.join(os.getcwd(), 'lunar-lander'), video_callable=lambda episode_id: True, force=True)

    for i in range(n_games):
        done = False
        score = 0
        observation = env.reset()
        while not done:
            action = ddqn_agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            ddqn_agent.remember(observation, action, reward, observation_, done)
            observation = observation_
            ddqn_agent.learn()
            env.render()

        eps_history.append(ddqn_agent.epsilon)
        ddqn_scores.append(score)

        avg_score = np.mean(ddqn_scores[max(0, i-100):(i+1)])
        epsilon = ddqn_agent.get_epsilon()
        print('episode ', i, ' | score %.2f' % score,
              ' | average score %.2f' % avg_score, ' | epsilon %.2f' % epsilon)

        if i % 10 == 0 and i > 0:
            ddqn_agent.save_model()
        if i == n_games-1:
            ddqn_agent.save_model()

    filename = 'lunarlander-ddqn.png'
    x = [i+1 for i in range(n_games)]
    plotLearning(x, ddqn_scores, eps_history, filename)