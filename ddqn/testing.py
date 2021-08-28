import numpy as np
import gym
import os
from datetime import datetime
from tensorflow.keras.models import load_model


def choose_action(q_net, state):
    state = state[np.newaxis, :]
    actions = q_net.predict(state)
    act = np.argmax(actions)
    return act


if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    # dqn_models_folder_list = []
    # dqn_models_folder_path = []
    # for folder in os.listdir(os.getcwd() + '//dqn_models'):
    #     date_time_str = folder.split("_", 1)[1]
    #     date_time_obj = datetime.strptime(date_time_str, '%Y-%m-%d_%H-%M-%S')
    #     dqn_models_folder_list.append(date_time_obj)
    #     dqn_models_folder_path.append(os.getcwd() + '//dqn_models' + '//' + folder)
    # latest_folder = max(dqn_models_folder_list)
    # latest_folder_index = dqn_models_folder_list.index(latest_folder)
    # model_file = dqn_models_folder_path[latest_folder_index] + "//dqn_model.h5"
    # model = load_model(model_file)
    model = load_model('ddqn_model.h5')
    n_games = 50
    for i in range(n_games):
        done = False
        observation = env.reset()
        while not done:
            action = choose_action(model, observation)
            observation_, reward, done, info = env.step(action)
            env.render()
            observation = observation_
