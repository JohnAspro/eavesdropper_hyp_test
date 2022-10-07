from env import *
from eval_env import *
from chernoff_test import *
from stable_baselines3 import A2C,PPO,DQN
import os

models_dir = "models"
log_dir = "logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

hor = 25
a = 1
b = 1

# RL agent
env=evasive_hypothesis_testing_env(hor, p_a_h, q_a_h, a, b)
Eval_env = eval_env_AHT(hor, p_a_h, q_a_h, a, b)
model = PPO('MlpPolicy',env, verbose = 1, tensorboard_log = log_dir, create_eval_env = True)

for i in range(1,30):
    model.learn(total_timesteps = 10000, reset_num_timesteps = False, \
        tb_log_name="PPO", eval_env = Eval_env, eval_freq = 1000, n_eval_episodes=1000)
    model.save(f"{models_dir}/{10000*i}")

# Chernoff
chernoff_eval_env = eval_env_AHT(hor,p_a_h,q_a_h,a,b)

episodes = 20
for episode in range(1, episodes+1):
    state = Eval_env.reset()
    chernoff_eval_env.reset()
    done = False
    score = 0 
    
    while not done:
        #env.render()lamda: env
        action,_states = model.predict(state)
        action=chernoffStrategy(chernoff_eval_env.legit_belief_vector)
        chern_state, cher_reward, done, c_info = chernoff_eval_env.step(action)
        state, RL_reward, done, info = env.step(action)
    print('Episode {} : RL Agent got {} and Chernoff Test gave {}'.format(episode, reward,))