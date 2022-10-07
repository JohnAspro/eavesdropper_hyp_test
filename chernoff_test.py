from env import *

def get_prob_vector(n):
    xs=np.linspace(0,1,num=n)
    vectors=[]
    for x in xs:
        vectors.append(tuple((x,1-x)))
    return vectors

def getKLDiv(p,q):
  res=0.0
  for i in range(len(p)):
    res=res+p[i]*np.log2(p[i]/q[i])
  return res
def chernoffStrategy(ro):
  g_s=get_prob_vector(100)
  g_vals=[]
  j_hat=np.argmax(ro)
  for g in g_s:
    g_val=0.0
    for a in [0,1]:
      j_vals=[]
      for j in [0,1,2]:
        if j==j_hat: 
          continue
        j_vals.append(g[a]*getKLDiv(p_a_h[a][j_hat],p_a_h[a][j]))
      g_val=min(j_vals)
    g_vals.append(g_val)
  index=np.argmax(g_vals)
  bestG=g_s[index]
  action= np.random.choice([0,1],p=bestG)
  return action

# test_env_data=[]
# hor = 25
# a = 1
# b = 1
# env=evasive_hypothesis_testing_env(hor, p_a_h, q_a_h, a, b)
# test_episodes=1000
# errorProbs=0.0
# random_times=[2^l for l in range(10)]
# for train_episode in range(test_episodes):
#   env.reset()
  
#   for t in range(env.horizon):
#     #print(env.ro)
    
#     if t in random_times:
#       action=np.random.choice([0,1])
#     else:
#       action=chernoffStrategy(env.legit_belief_vector)
#     state,y,done,info=env.step(action)
#     # print(env.hypothesis,"action,ro=",action,"-",env.legit_belief_vector)
#     if done==True:
#       errorProbs+=1-max(env.legit_belief_vector)
# print(errorProbs/test_episodes)
  