import numpy as np
import matplotlib.pyplot as plt
import math
from simulation_functions import create_watermaze, create_actions, bounce_off, goal_not_reached
from create_figure3 import figure3, plot_Cp
from actorcritic_functions import place_cells, critic

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

# initialize weights to train and vectors of activities:
w = np.random.rand(N)
z = np.random.rand(8, N)
f, places = place_cells([0.9, 0.001])
C = np.zeros(N)
a = np.zeros((8, N))
places_x, places_y = np.transpose(places)

#C = critic(w, f)
# a = actor(z, f)
print "plot!"
plot_Cp(w, places_x, places_y)


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

        else: # goal reached
            print "goal reached!"
            break
#
# plt.plot(x, y)
# plt.show()