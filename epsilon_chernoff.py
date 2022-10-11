from chernoff_test import *
import numpy as np
from eval_env import *
def inv_chernoff(ro): 
    g_s=get_prob_vector(100)
    g_vals=[]
    j_hat=np.argmax(ro)

    for g in g_s:
        g_val=0.0
        j_vals=[]
        for j in [0,1,2,3]:
            if j==j_hat:
                continue
            j_val=0.0
            for a in [0,1,2]:

                j_val+=g[a]*getKLDiv(p_a_h[a][j_hat],p_a_h[a][j])
            j_vals.append(j_val)
        g_vals.append(max(j_vals))
    index=np.argmin(g_vals)
    bestG=g_s[index]
    action=np.random.choice([0,1,2],p=bestG)
    return action
def epsilon_chernoff(ro,ro_adv,epsilon=0.5):
    if random.uniform(0,1)<=epsilon:
        return chernoffStrategy(ro)
    else:
        return inv_chernoff(ro_adv)
         
# hor = 100
# a = 1
# b = 1
# Eval_env = eval_env_AHT(hor, p_a_h, q_a_h, a, b)
# testEpisodes=1
# rew=0.0
# uni_times=[2**l for l in range(10)]
# for te in range(testEpisodes):
#   obs=Eval_env.reset(H=0)
#   for t in range(Eval_env.horizon):
#     #if t in uni_times:
#     #  action=random.randint(0,2)
#     #else:
#     action=naive_strategy(Eval_env.legit_belief_vector,Eval_env.adv_belief_vector,a,b)
#     state,reward,done,info=Eval_env.step(action)
#     print("legit error prob=",1-max(Eval_env.legit_belief_vector),"eave error prob=",1-max(Eval_env.adv_belief_vector))
#     if done==True:
#       rew+=reward
#       print("done")
#       break 
# print(rew/testEpisodes)
        