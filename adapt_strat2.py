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
def update_b_v(a,y,z,lbv,abv):
		l_sigma = 0.0
		a_sigma	 = 0.0
		lbv_new = []
		abv_new = []
		for i in [0,1,2,3]:
			l_sigma += lbv[i]*p_a_h[a][i][y] 
			a_sigma += abv[i]*q_a_h[a][i][z]
		for i in [0,1,2,3]:
			lbv_new.append(lbv[i]*p_a_h[a][i][y]/l_sigma)
			abv_new.append(abv[i]*q_a_h[a][i][z]/a_sigma)
		return lbv_new, abv_new
#deterministically decides on the action a that maxmizes the one step expectation of 
# -a *ler+b*aer
def naive_strategy(ro,ro_adv,alpha,beta):
    alpha_scores=[]
    for a in [0,1,2]:
        alpha_score=0.0
        for y in [0,1]:
            for z in [0,1]:
                #first compute sigma
                sigma=0.0
                for j in [0,1,2,3]:
                    sigma+=ro[j]*p_a_h[a][j][y]
                z_sigma=0.0
                for j in [0,1,2,3]:
                    z_sigma+=ro[j]*q_a_h[a][j][z] 
                
                score=0.0
                for j  in [0,1,2,3]:
                    ro_new,ro_adv_new = update_b_v(a,y,z,ro,ro_adv)
                    score+=ro[j]*( (1-max(ro_new))*(-alpha)+(1-max(ro_adv_new))*beta )
                alpha_score+=score*sigma*z_sigma
        alpha_scores.append(alpha_score)
    action=np.argmax(alpha_scores)
    return action
def get_confidence(ro):
    c=0.0
    for i in [0,1,2,3]:
        if ro[i]-1==0:
            c+=99999
        else:
            c+=ro[i]*np.log2(ro[i]/1-ro[i])
    return c
def adaptive_conf_strategy(ro,ro_adv,alpha,beta):
    alpha_scores=[]
    for a in [0,1,2]:
        alpha_score=0.0
        for y in [0,1]:
            for z in [0,1]:
                #first compute sigma
                sigma=0.0
                for j in [0,1,2,3]:
                    sigma+=ro[j]*p_a_h[a][j][y]
                z_sigma=0.0
                for j in [0,1,2,3]:
                    z_sigma+=ro[j]*q_a_h[a][j][z] 
                
                score=0.0
                ro_new,ro_adv_new=update_b_v(a,y,z,ro,ro_adv)
                score=alpha*get_confidence(ro_new)-beta*get_confidence(ro_adv_new)
                alpha_score+=score*sigma*z_sigma
        alpha_scores.append(alpha_score)
    action=np.argmax(alpha_scores)
    return action
    

# hor = 100
# a = 1
# b = 1
# Eval_env = eval_env_AHT(hor, p_a_h, q_a_h, a, b)
# testEpisodes=1
# rew=0.0
# uni_times=[2**l for l in range(10)]
# for te in range(testEpisodes):
#   obs=Eval_env.reset()
#   for t in range(Eval_env.horizon):
#     #if t in uni_times:
#     #  action=random.randint(0,2)
#     #else:
#     action=adaptive_conf_strategy(Eval_env.legit_belief_vector,Eval_env.adv_belief_vector,a,b)
#     state,reward,done,info=Eval_env.step(action)
#     print("legit error prob=",1-max(Eval_env.legit_belief_vector),"eave error prob=",1-max(Eval_env.adv_belief_vector))
#     if done==True:
#       rew+=reward
#       print("done")
#       break 
# print(rew/testEpisodes)
        