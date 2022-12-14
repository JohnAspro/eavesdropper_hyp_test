from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
import random

p_a_h={}
p_a_h[0]={}
p_a_h[1]={}
p_a_h[2]={}

p_a_h[0][0]=[0.78,0.22]
p_a_h[0][1]=[0.2,0.8]
p_a_h[0][2]=[0.82,0.18]
p_a_h[0][3]=[0.83,0.17]

p_a_h[1][0]=[0.79,0.21]
p_a_h[1][1]=[0.8,0.2]
p_a_h[1][2]=[0.19,0.81]
p_a_h[1][3]=[0.77,0.23]

p_a_h[2][0]=[0.75,0.25]
p_a_h[2][1]=[0.84,0.16]
p_a_h[2][2]=[0.81,0.19]
p_a_h[2][3]=[0.3,0.7]


q_a_h={}
q_a_h[0]={}
q_a_h[1]={}
q_a_h[2]={}

q_a_h[0][0]=[0.67,0.33]
q_a_h[0][1]=[0.33,0.67]
q_a_h[0][2]=[0.69,0.31]
q_a_h[0][3]=[0.7,0.3]

q_a_h[1][0]=[0.7,0.3]
q_a_h[1][1]=[0.68,0.32]
q_a_h[1][2]=[0.35,0.65]
q_a_h[1][3]=[0.67,0.33]

q_a_h[2][0]=[0.65,0.35]
q_a_h[2][1]=[0.74,0.26]
q_a_h[2][2]=[0.79,0.21]
q_a_h[2][3]=[0.35,0.65]


# 3 sensors with fixed distributions for the model and the adversary obs
# 4 hypothesis -> all normal , first, second or third behaving abnormally
class eval_env_AHT(Env):		 
	def __init__(self,horizon,p_a_h, q_a_h, a, b):
		# actions either pick first or second sensor
		self.action_space = Discrete(3)
		#uniform prior belief vector values
		self.prior = np.array([1/4, 1/4, 1/4, 1/4])
		#agent only sees the legit belief vector
		self.observation_space = \
			Box(low = np.array([0.0,0.0,0.0,0.0]), high = np.array([1.0,1.0,1.0,1.0]), dtype = np.float64)
		self.horizon = horizon
		self.p_a_h = p_a_h
		self.q_a_h = q_a_h
		self.a = a
		self.b = b

	def update_b_v(self,a,y,z,lbv,abv):
		l_sigma = 0.0;
		a_sigma	 = 0.0;
		lbv_new = []
		abv_new = []
		for i in [0,1,2,3]:
			l_sigma += lbv[i]*self.p_a_h[a][i][y] 
			a_sigma += abv[i]*self.q_a_h[a][i][z]
		for i in [0,1,2,3]:
			lbv_new.append(lbv[i]*self.p_a_h[a][i][y]/l_sigma)
			abv_new.append(abv[i]*self.q_a_h[a][i][z]/a_sigma)
		return lbv_new, abv_new

	def step(self,action):
		done = False
		if self.t == self.horizon - 1:
			done = True
		self.t += 1

		y = np.random.choice(np.array([0,1]), p = self.p_a_h[action][self.hypothesis])
		z = np.random.choice(np.array([0,1]), p = self.q_a_h[action][self.hypothesis])
		
		self.legit_belief_vector, self.adv_belief_vector = \
			self.update_b_v(action, y, z, self.legit_belief_vector, self.adv_belief_vector)
		#legitimate errors probability
		self.ler = 1 - np.amax(self.legit_belief_vector)
		#adversary errors probability
		self.aer = 1 - np.amax(self.adv_belief_vector) 
		
		if done:
			# reward = self.a*np.amax(self.adv_belief_vector)/(self.b*np.amax(self.legit_belief_vector))
			reward = -self.a*self.ler + self.b*self.aer
		else:
			reward = 0
		
		info = {}

		return self.legit_belief_vector, reward, done, info
	
	def render(self):
		pass

	def reset(self):
		#init the hypothesis and the belief vectors
		self.hypothesis = random.randint(0,3)
		# self.hypothesis = 0
		self.legit_belief_vector = self.prior
		self.adv_belief_vector = self.prior
		self.t = 0
		self.ler = 0.75
		self.aer = 0.75
		return self.legit_belief_vector

