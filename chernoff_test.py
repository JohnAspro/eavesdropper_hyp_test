from env import *
from eval_env import *

def get_prob_vector(n):
    xs=np.linspace(0,1,num=n)
    vectors=[]
    for x in xs:      
        for y in xs:
            if x+y>1:
                continue
            new_v=[]
            new_v.append(x)
            new_v.append(y)
            new_v.append(1-(x+y))
            vectors.append(new_v)
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
    for a in [0,1,2]:
      j_vals=[]
      for j in [0,1,2,3]:
        if j==j_hat: 
          continue
        j_vals.append(g[a]*getKLDiv(p_a_h[a][j_hat],p_a_h[a][j]))
      g_val=min(j_vals)
    g_vals.append(g_val)
  index=np.argmax(g_vals)
  bestG=g_s[index]
  action= np.random.choice([0,1,2],p=bestG)
  return action  
hor = 100
a = 1
b = 1
Eval_env = eval_env_AHT(hor, p_a_h, q_a_h, a, b)
testEpisodes=1
rew=0.0
uni_times=[2**l for l in range(10)]
for te in range(testEpisodes):
  obs=Eval_env.reset(H=0)
  for t in range(Eval_env.horizon):
    if t in uni_times:
      action=random.randint(0,2)
    else:
      action=chernoffStrategy(Eval_env.legit_belief_vector)
    state,reward,done,info=Eval_env.step(action)
    print("legit error prob=",1-max(Eval_env.legit_belief_vector),"eave error prob=",1-max(Eval_env.adv_belief_vector))
    if done==True:
      rew+=reward
      break 
print(rew/testEpisodes)