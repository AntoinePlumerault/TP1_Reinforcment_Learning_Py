from random import random
import numpy as np



def choose(pi):
	cum_pi = np.cumsum(pi)
	r = random()
	for i in range(len(pi)):
		if r < cum_pi[i]:
			return i  

class MDP:
	def __init__(self, P_x_a, R_x_a):
		self.X = ['x0', 'x1', 'x2']
		self.P_x_a = P_x_a
		self.R_x_a = R_x_a
		self.n_states = len(P_x_a[0])
		self.n_actions = len(P_x_a)
	
	def reset(self, pi):
		assert len(pi) == self.S
		assert abs(sum(pi)-1) < 1e-5
		x0 = choose(pi)
		return x0

	def step(self, state, action):
		p = self.P_x_a[action, state]
		next_state = choose(p)
		reward = self.R_x_a[action, state]
		return next_state, reward