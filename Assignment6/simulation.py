import numpy as np
import matplotlib.pyplot as plt
import math

np.random.seed(5)

# function to initialize the water maze:
def create_watermaze():
    r = 1.0
    phi = np.arange(0, 2.0 * np.pi, 0.03)
    xaxis = r*np.cos(phi)
    yaxis = r*np.sin(phi)
    goal = np.random.randint(0, 210)
    r = round(np.random.uniform(0.0, 1.0), 2)
    return xaxis, yaxis, xaxis[goal]*r, yaxis[goal]*r


# function to create all different actions/directions:
def create_actions():
    actions_x = np.zeros(8)
    actions_y = np.zeros(8)
    for action in range(8):
        actions_x[action] = np.cos(action * np.pi * 0.25 + np.pi)
        actions_y[action] = np.sin(action * np.pi * 0.25 + np.pi)
    return actions_x, actions_y


# perform bounce-off and return new locations:
def bounce_off(x, y, dx, dy):
    angle = np.arctan2(y + dy, x + dx)
    angle_step = np.arctan2(dy, dx)

    # compute new angle:
    angle_new = 2.0 * angle - np.pi - angle_step
    dx_new = np.cos(angle_new)
    dy_new = np.sin(angle_new)

    # compute bounce-off locations:
    x_new = np.cos(angle) + 0.03 * dx_new * np.linalg.norm([x + dx, y + dy], ord=2)
    y_new = np.sin(angle) + 0.03 * dy_new * np.linalg.norm([x + dx, y + dy], ord=2)
    return x_new, y_new


# function that determines if the escape platform is (not) reached:
def goal_not_reached(x, y, goalx, goaly):
    distance = math.sqrt(np.power(x - goalx, 2)) + math.sqrt(np.power(y - goaly, 2))
    return distance >= 0.05


# create a water maze and plot:
x_coords, y_coords, goal_x, goal_y = create_watermaze()
plt.figure(0)
plt.plot(x_coords, y_coords, c='black', ls='-')
plt.plot(goal_x, goal_y, c='black', fillstyle='none', marker='o')

# initialize parameters:
dt = 0.1
T = 3000.0

# initial location:
x = [x_coords[100]]
y = [y_coords[100]]
actionsx, actionsy = create_actions()

for t in range(int(T / dt)):
    if goal_not_reached(x[-1], y[-1], goal_x, goal_y):
        direction = np.random.randint(0, 8)
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

plt.plot(x, y)
plt.show()