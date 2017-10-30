from mdp import MDP
import numpy as np
import matplotlib.pyplot as plt
import time

P_x_a = np.array([[[0.45, 0.55, 0.00],
		           [0.60, 0.40, 0.00],
		           [0.00, 1.00, 0.00]],
	              [[0.00, 1.00, 0.00],
	               [0.00, 0.10, 0.90],
	               [0.50, 0.10, 0.40]]])

R_x_a = np.array([[-0.4, -1.0, 2.0], 
	              [ 0.0, -0.5, 0.0]])

env = MDP(P_x_a, R_x_a)

X = range(env.n_states)
A = range(env.n_actions)
gamma = 0.95

# ========= Policy evaluation ========= #

pi_opt = [1,1,0]  # Optimal policy

V_opt = np.array([0.0] * env.n_states)
T = 10000
for t in range(T):
	for x in X:
		next_state, reward = env.step(x, pi_opt[x])
		V_opt[x] = reward + gamma*sum(P_x_a[pi_opt[x], x, :] * V_opt[:]) # WARNING 


# ========== Value iteration ========== #
t = time.time()

Delta_V_Value = []

V = np.array([0.0] * env.n_states)
term = False
while not term:
	TV = np.array([0.0] * env.n_states)
	for x in X:
		TV[x] = -float('inf')
		for a in A:
			next_state, reward = env.step(x, a)
			TVa = reward + gamma*sum(P_x_a[a, x, :] * V[:])
			if TVa > TV[x]:
				TV[x] = TVa

	#print(max(abs(TV - V)))
	if max(abs(TV - V)) < 0.01:
		term = True

	Delta_V_Value.append(max(abs(V_opt - V)))

	V = TV

pi = [A[0]] * env.n_states
for x in X:
	best_a = A[0]
	best_Va = -float('inf')
	for a in A:
		next_state, reward = env.step(x, a)
		Va = reward + gamma*sum(P_x_a[a, x, :] * V[:])
		if Va > best_Va:
			best_a = a
			best_Va = Va
	pi[x] = best_a

elapsed = time.time() - t

plt.plot(Delta_V_Value)

print('elapsed Value Iteration', elapsed)

# ========= Policy iteration ========== #

t = time.time()

Delta_V_Policy = []

pi = [A[0]] * env.n_states
V = np.array([0.0] * env.n_states)
term = False
while (not term):
	V_pi = np.array([0.0] * env.n_states)
	for x in X:
		next_state, reward = env.step(x, pi[x])
		V_pi[x] = reward + gamma*sum(P_x_a[pi[x], x, :] * V[:])

	for x in X:
		best_a = A[0]
		best_Va = -float('inf')
		for a in A:
			next_state, reward = env.step(x, a)
			Va = reward + gamma*sum(P_x_a[a, x, :] * V[:])
			if Va > best_Va:
				best_a = a
				best_Va = Va
		pi[x] = best_a
	
	Delta_V_Policy.append(max(abs(V_opt - V)))
	
	if max(abs(V_pi - V)) < 0.01:
		term = True

	V = V_pi

elapsed = time.time() - t

plt.plot(Delta_V_Policy)

print(len(Delta_V_Policy),len(Delta_V_Value))
print('elapsed Policy Iteration', elapsed)

# ====================================== #

plt.show()

print(2*0.01 * gamma / (1.0-gamma))
print(pi)
print(V)
print(V_opt)



