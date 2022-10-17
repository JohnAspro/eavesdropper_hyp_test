from chernoff_test import *
from epsilon_chernoff import *
from naive_strategy import *
from test_f_env import *
from stable_baselines3 import PPO,DQN
import os
import matplotlib.pyplot as plt
import numpy as np

hor = 50
a = 1
b = 1
episodes = 10
timespace = np.arange(0,hor,1)

# RL agent
m_env = test_f_env_AHT(hor, a, b)
models_dir = "models"
RL_algorithm = "DQN" 
model_path = f"{models_dir}/{RL_algorithm}/{RL_algorithm}_f_balanced/160000.zip"
model = DQN.load(model_path, m_env)
m_av_err = [[0]*hor,[0]*hor]

# Chernoff strategy
c_env = test_f_env_AHT(hor,a,b)
c_av_err = [[0]*hor,[0]*hor]

#epsilon chernoff strategy
eps_c_env = test_f_env_AHT(hor,a,b)
eps_c_av_err = [[0]*hor,[0]*hor]

#naive strategy
n_env = test_f_env_AHT(hor,a,b)
n_av_err = [[0]*hor,[0]*hor]

for episode in range(1, episodes+1):
    state = m_env.reset()
    chern_state = c_env.reset()
    eps_c_state = eps_c_env.reset()
    n_state = n_env.reset()
    done = False
    
    # print(m_env.hypothesis, c_env.hypothesis, eps_c_env.hypothesis, n_env.hypothesis)
    print(episode)

    while not done:
        m_av_err[0][m_env.t] += m_env.ler/episodes 
        m_av_err[1][m_env.t] += m_env.aer/episodes
        c_av_err[0][c_env.t] += c_env.ler/episodes
        c_av_err[1][c_env.t] += c_env.aer/episodes
        eps_c_av_err[0][eps_c_env.t] += eps_c_env.ler/episodes
        eps_c_av_err[1][eps_c_env.t] += eps_c_env.aer/episodes
        n_av_err[0][n_env.t] += n_env.ler/episodes
        n_av_err[1][n_env.t] += n_env.aer/episodes
       
        #model action
        action,_states = model.predict(state)
        state, RL_reward, done, info = m_env.step(int(action))

        #classic chernoff action
        action=chernoffStrategy(chern_state[1])
        chern_state, cher_reward, done, c_info = c_env.step(action)

        #epsilon chernoff action
        action = epsilon_chernoff(eps_c_state[0],eps_c_state[1], epsilon = 0.5)
        eps_c_state, eps_cher_reward, done, eps_c_info = eps_c_env.step(action)

        #naive strategy action
        action = naive_strategy(n_state[0],n_state[1],a,b)
        n_state, n_reward, done, n_info = n_env.step(action)

plt.title('Error Averaged over {} episodes'.format(episodes))
plt.ylabel('Legitimate Error Probability')
plt.xlabel('Steps')
plt.plot(timespace, m_av_err[0], '-g', label = RL_algorithm)
plt.plot(timespace, c_av_err[0], '-b', label = 'Chernoff')
plt.plot(timespace, eps_c_av_err[0], '-r', label = 'Epsilon Chernoff')
plt.plot(timespace, n_av_err[0], '-m', label = 'Naive Strategy')
plt.legend()
plt.grid()
plt.show()   
plt.title('Error Averaged over {} episodes'.format(episodes))
plt.ylabel('Adversary Error Probability')
plt.xlabel('Steps')
plt.plot(timespace, m_av_err[1], '-g', label = RL_algorithm)
plt.plot(timespace, c_av_err[1], '-b', label = 'Chernoff')
plt.plot(timespace, eps_c_av_err[1], '-r', label = 'Epsilon Chernoff')
plt.plot(timespace, n_av_err[1], '-m', label = 'Naive Strategy')
plt.legend()
plt.grid()    
plt.show() 
#     print('Episode {}, hyp {}: RL Agent got {} and Chernoff Test gave {}'.format(episode, env.hypothesis, RL_reward, cher_reward))
# print('total average score for RL {} and for chernoff {} '.format(RL_score/episodes, cher_score/episodes))
