import numpy as np
import matplotlib.pyplot as plt
import math
from simulation_functions import create_watermaze, create_actions, bounce_off, goal_not_reached
from create_figure3 import figure3, plot_Cp
from actorcritic_functions import place_cells, critic, actor, delta

np.random.seed(40)

# create a water maze and plot:
x_coords, y_coords, goal_x, goal_y = create_watermaze()
plt.figure(0)
plt.plot(x_coords, y_coords, c='black', ls='-')
plt.plot(goal_x, goal_y, c='black', fillstyle='none', marker='o')
actionsx, actionsy = create_actions()
N = 493

# initialize parameters:
dt = 0.1
T = 120.0
max_trials = 25
discount = 0.9

# initialize weights to train and vectors of activities:
w = np.zeros((max_trials, N))
z = np.zeros((max_trials, 8, N))
f, places = place_cells([0.9, 0.001])
C = np.zeros(max_trials) # N?
places_x, places_y = np.transpose(places)
x_path = np.zeros((max_trials, int(T / dt)))
y_path = np.zeros((max_trials, int(T / dt)))

for trial in range(max_trials):
    # initial location and direction:
    x = [x_coords[int(209 / np.random.randint(1, 4))]]
    y = [y_coords[int(209 / np.random.randint(1, 4))]]
    direction = np.random.randint(0, 8)
    for t in range(int(T / dt)):
        if goal_not_reached(x[-1], y[-1], goal_x, goal_y):
            new_direction = np.random.randint(0, 28)
            if new_direction >= 8:
                new_direction = direction
            direction = new_direction
            dx = actionsx[direction] * 0.03
            dy = actionsy[direction] * 0.03

            # check for boundaries:
            if np.linalg.norm([x[-1] + dx, y[-1] + dy], ord=2) > 1.0:
                # bounce off:
                xnew, ynew = bounce_off(x[-1], y[-1], dx, dy)
                x.append(xnew)
                y.append(ynew)
            else:
                x.append(x[-1] + dx)
                y.append(y[-1] + dy)

            # compute current C:
                f_p, _ = place_cells([x[-1], y[-1]])
                C_current = critic(w[trial, :], f_p)

            # compute action probabilities and perform best action:
                a = actor(z[trial, :, :], f_p) # 8x1
                P = np.exp(2.0 * a) / np.sum(np.exp(2.0 * a))
                print "P: ", np.shape(P)


            # the critic (update w):
                f_p_new, _ = place_cells([x[-1], y[-1]])
                C_new = critic(w[trial, :], f_p_new)
                dw = np.repeat(delta(C_current, C_new, discount, goal_not_reached(x[-2], y[-2], goal_x, goal_y)), N) * f_p
                w[trial + 1, :] = w[trial, :] + dw

            # the actor (update z):

        else: # goal reached
            print "goal reached!"
            break
#
# plt.plot(x, y)
# plt.show()