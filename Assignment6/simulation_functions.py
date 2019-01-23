import numpy as np
import matplotlib.pyplot as plt
import math


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
