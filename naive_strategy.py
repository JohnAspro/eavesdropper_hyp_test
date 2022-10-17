from eval_f_env import *

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
                    score+=ro[j]*((1-max(ro_new))*(-alpha)+(1-max(ro_adv_new))*beta)
                alpha_score+=score*sigma*z_sigma
        alpha_scores.append(alpha_score)
    action=np.argmax(alpha_scores)
    return action