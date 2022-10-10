from env import *
from eval_env import *
from chernoff_test import *
from stable_baselines3 import A2C,PPO,DQN
import os
import matplotlib.pyplot as plt
import numpy as np

hor = 100
a = 1
b = 1

# RL agent
env = eval_env_AHT(hor, p_a_h, q_a_h, a, b)
models_dir = "models"
model_path = f"{models_dir}/PPO/280000.zip"
model = PPO.load(model_path, env)

# Chernoff
chernoff_eval_env = eval_env_AHT(hor,p_a_h,q_a_h,a,b)

episodes = 1
RL_score = 0 
cher_score = 0 
model_ler = []
model_aer = []
chern_ler = []
chern_aer = []
timespace = np.arange(0,hor,1)

for episode in range(1, episodes+1):
    state = env.reset()
    chern_state = chernoff_eval_env.reset()
    done = False
    print(env.hypothesis, chernoff_eval_env.hypothesis)
    while not done:
        action,_states = model.predict(state)
        state, RL_reward, done, info = env.step(action)

        action=chernoffStrategy(chern_state)
        chern_state, cher_reward, done, c_info = chernoff_eval_env.step(action)
        print("LER RL", env.ler, "Chernoff", chernoff_eval_env.ler, "AEP RL", env.aer, "Chernoff", chernoff_eval_env.aer)
        model_ler.append(env.ler)
        model_aer.append(env.aer)
        chern_ler.append(chernoff_eval_env.ler)
        chern_aer.append(chernoff_eval_env.aer)
    print(len(timespace))
    plt.plot(timespace, model_ler, timespace, chern_ler)
    plt.ylabel('Legitimate Error Probability')
    plt.grid()    
    plt.show()
    plt.plot(timespace, model_aer, timespace, chern_aer)
    plt.ylabel('Adversary Error Probability')
    plt.grid()
    plt.show()
    RL_score += RL_reward
    cher_score += cher_reward     
    print(env.legit_belief_vector)
    print(chernoff_eval_env.legit_belief_vector)
#     print('Episode {}, hyp {}: RL Agent got {} and Chernoff Test gave {}'.format(episode, env.hypothesis, RL_reward, cher_reward))
# print('total average score for RL {} and for chernoff {} '.format(RL_score/episodes, cher_score/episodes))
