from env import *
from eval_env import *
from chernoff_test import *
from stable_baselines3 import PPO,DQN
from eval_f_env import *
from f_env import *
import os

models_dir = "models"
log_dir = "logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

hor = 50
a = 0.8
b = 1.2

# RL agent
env=f_env_AHT(hor, a, b)
Eval_env = eval_f_env_AHT(hor)
model = PPO('MlpPolicy', env, verbose = 1, tensorboard_log = log_dir, create_eval_env = True)

for i in range(1,100):
    model.learn(total_timesteps = 10000, reset_num_timesteps = False, \
        tb_log_name="DQN2", eval_env = Eval_env, eval_freq = 10000, n_eval_episodes=1000)
    if i % 2 == 0:
        model.save(f"{models_dir}/PPO/PPO_f_balanced/{10000*i}")