from env import *
from eval_env import *
from chernoff_test import *
from stable_baselines3 import A2C,PPO,DQN
import os

hor = 25
a = 1
b = 1

# RL agent
env = eval_env_AHT(hor, p_a_h, q_a_h, a, b)
models_dir = "models"
model_path = f"{models_dir}/PPO/280000.zip"
model = PPO.load(model_path, env)

# Chernoff
chernoff_eval_env = eval_env_AHT(hor,p_a_h,q_a_h,a,b)

episodes = 100
RL_score = 0 
cher_score = 0 

for episode in range(1, episodes+1):
    state = env.reset()
    chernoff_eval_env.reset()
    done = False

    while not done:
        action,_states = model.predict(state)
        state, RL_reward, done, info = env.step(action)
        
        action=chernoffStrategy(chernoff_eval_env.legit_belief_vector)
        chern_state, cher_reward, done, c_info = chernoff_eval_env.step(action)
    RL_score += RL_reward
    cher_score += cher_reward     
    print('Episode {}, hyp {}: RL Agent got {} and Chernoff Test gave {}'.format(episode, env.hypothesis, RL_reward, cher_reward))
print('total average score for RL {} and for chernoff {} '.format(RL_score/episodes, cher_score/episodes))