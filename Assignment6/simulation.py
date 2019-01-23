import numpy as np
import matplotlib.pyplot as plt

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


# check if new location falls within maze and perform bounce-off:
def check_boundaries(x, y, dx, dy):
    if np.linalg.norm([x + dx, y + dy], ord=2) > 1.0:
        return False
    else:
        return True


# create a water maze and plot:
x_coords, y_coords, goal_x, goal_y = create_watermaze()
plt.figure(0)
plt.plot(x_coords, y_coords, c='black', ls='-')
plt.plot(goal_x, goal_y, c='black', fillstyle='none', marker='o')

# initialize parameters:
dt = 0.1
T = 30.0

# initial location:
x = [x_coords[100]]
y = [y_coords[100]]
actionsx, actionsy = create_actions()

for t in range(int(T / dt)):
    direction = np.random.randint(0, 8)
    dx = actionsx[direction] * 0.03
    dy = actionsy[direction] * 0.03
    if check_boundaries(x[-1], y[-1], dx, dy):
        x.append(x[-1] + dx)
        y.append(y[-1] + dy)
    print x[-1]
    print y[-1]

plt.plot(x, y)




plt.show()