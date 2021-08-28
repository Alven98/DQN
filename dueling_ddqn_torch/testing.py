import gym
from dueling_ddqn_torch import Agent


if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    agent = Agent(gamma=0.99, epsilon=0.01, eps_min=0.01, eps_dec=1e-3,
                  lr=5e-4, input_dims=[8], n_actions=4, mem_size=1000000,
                  batch_size=64, replace=100)
    agent.load_models()
    n_games = 50
    for i in range(n_games):
        done = False
        observation = env.reset()
        while not done:
            action = agent.testing(observation)
            observation_, reward, done, info = env.step(action)
            env.render()
            observation = observation_
