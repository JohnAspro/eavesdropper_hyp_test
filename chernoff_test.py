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