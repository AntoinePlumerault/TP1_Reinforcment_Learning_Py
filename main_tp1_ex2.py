from gridworld import GridWorld1
import gridrender as gui
import numpy as np
import time
import matplotlib.pyplot as plt

env = GridWorld1

################################################################################
# investigate the structure of the environment
# - env.n_states: the number of states
# - env.state2coord: converts state number to coordinates (row, col)
# - env.coord2state: converts coordinates (row, col) into state number
# - env.action_names: converts action number [0,3] into a named action
# - env.state_actions: for each state stores the action availables
#   For example
#       print(env.state_actions[4]) -> [1,3]
#       print(env.action_names[env.state_actions[4]]) -> ['down' 'up']
# - env.gamma: discount factor
################################################################################
print(env.state2coord)
print(env.coord2state)
print(env.state_actions[4])
print(env.action_names[env.state_actions[4]])

################################################################################
# Policy definition
# If you want to represent deterministic action you can just use the number of
# the action. Recall that in the terminal states only action 0 (right) is
# defined.
# In this case, you can use gui.renderpol to visualize the policy
################################################################################

pol = [1, 2, 0, 0, 1, 1, 0, 0, 0, 0, 3]
gui.render_policy(env, pol)

################################################################################
# Try to simulate a trajectory
# you can use env.step(s,a, render=True) to visualize the transition

env.render = True
state = 0
fps = 1
for i in range(5):
    action = np.random.choice(env.state_actions[state])
    nexts, reward, term = env.step(state,action)
    state = nexts
    time.sleep(1./fps)

################################################################################s
# You can also visualize the q-function using render_q
################################################################################
# first get the maximum number of actions available
max_act = max(map(len, env.state_actions))
q = np.random.rand(env.n_states, max_act)
gui.render_q(env, q)

################################################################################
# Work to do: Q4
################################################################################
# here the v-function and q-function to be used for question 4
v_q4 = [0.87691855, 0.92820033, 0.98817903, 0.00000000, 0.67106071, -0.99447514, 0.00000000, -0.82847001, -0.87691855,
        -0.93358351, -0.99447514]
q_q4 = [[0.87691855, 0.65706417],
        [0.92820033, 0.84364237],
        [0.98817903, -0.75639924, 0.89361129],
        [0.00000000],
        [-0.62503460, 0.67106071],
        [-0.99447514, -0.70433689, 0.75620264],
        [0.00000000],
        [-0.82847001, 0.49505225],
        [-0.87691855, -0.79703229],
        [-0.93358351, -0.84424050, -0.93896668],
        [-0.89268904, -0.99447514]
        ]

################################################################################
# Work to do: Q5
################################################################################
v_opt = [0.87691855, 0.92820033, 0.98817903, 0.00000000, 0.82369294, 0.92820033, 0.00000000, 0.77818504, 0.82369294,
         0.87691855, 0.82847001]

################################################################################
# ------------------ Question N°4 ------------------ #
env.render = False

J = []
for N in range(1, 100, 1):
    Q = np.array([0.0] * env.n_states)
    Tmax = 1000
    gamma = 0.95
    for n in range(N):
        for initial_state in range(env.n_states):
            #print(initial_state)
            state = initial_state
            for t in range(0,Tmax):
                if (0 in env.state_actions[state]):
                    action = 0
                else:
                    action = 3
                nexts, reward, term = env.step(state,action)
                Q[initial_state] += gamma**t * reward
                state = nexts
                if term: break
    Q /= N
    J.append((sum(Q,0) - sum(v_q4)) / 10)


print('Q estimation : ', Q)
# --------------------------------------------------- #

plt.plot(J)
plt.show()