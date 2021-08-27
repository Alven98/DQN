from dqn import Agent
import numpy as np
from utils import plotLearning
import gym

if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    n_games = 500
    agent = Agent(alpha=0.0005, gamma=0.99, n_actions=2, epsilon=1.0, input_dims=4,
                  mem_size=1000000, batch_size=64, epsilon_end=0.01, restart_training=True)

    scores = []
    eps_history = []

    for i in range(n_games):
        done = False
        score = 0
        observation = env.reset()
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            env.render()
            score += reward
            agent.remember(observation, action, reward, observation_, done)
            observation = observation_
            agent.learn()

        eps_history.append(agent.epsilon)
        scores.append(score)

        avg_score = np.mean(scores[max(0, i-100):(i+1)])
        epsilon = agent.get_epsilon()
        print('episode ', i, ' | score %.2f' % score,
              ' | average score %.2f' % avg_score, ' | epsilon %.2f' % epsilon)

        if i % 10 == 0 and i > 0:
            agent.save_model()
        if i == n_games-1:
            agent.save_model()

    filename = 'lunarlander.png'
    x = [i+1 for i in range(n_games)]
    plotLearning(x, scores, eps_history, filename)