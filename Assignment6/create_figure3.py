import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from actorcritic_functions import critic, place_cells
import itertools


# to make the 3D-plot of C(p) over all p:
def plot_Cp(w, x_coords, y_coords, fig):
    x_mesh, y_mesh = np.meshgrid(x_coords, y_coords)
    C = np.zeros((493, 493))
    for x in range(np.shape(x_mesh)[0]):
        a, b = place_cells([np.transpose(x_mesh)[x], y_mesh[x]])
        C[x, :] = critic(w, a)

    all_combinations = []
    for pair in itertools.product(np.linspace(-1.0, 1.0, 493), repeat=2):
        all_combinations.append(pair)

    comb_a, comb_b = np.transpose(all_combinations)
    distance = np.sqrt(np.power(comb_a, 2) + np.power(comb_b, 2))

    # make sure C becomes a circle:
    C = C.flatten(order='C')
    C[np.where(distance >= 1.0)] = np.nan
    C = np.reshape(C, (493, 493), order='C')

    # plot resulting C:
    #fig = plt.figure(1)
    ax = fig.add_subplot(111, projection='3d')
    #ax.plot_wireframe(x_mesh, y_mesh, C, rstride=1, cstride=1)
    ax.plot_surface(x_mesh, y_mesh, C, cmap=cm.viridis, rcount=493, ccount=493)
    ax.set_zlim(0.0, 1.0)
    plt.show()


def plot_maze(x_coords, y_coords, goal_x, goal_y, x, y):
    plt.plot(goal_x, goal_y, c='black', fillstyle='none', marker='o')
    plt.plot(x, y)
    plt.plot(x_coords, y_coords, c='black', ls='-')
    return 0


def plot_actions(x_coords, y_coords, goal_x, goal_y, z):
    plt.plot(goal_x, goal_y, c='black', fillstyle='none', marker='o')
    # compute actions and plot...
    x_mesh, y_mesh = np.meshgrid(x_coords, y_coords)

    plt.plot(x_coords, y_coords, c='black', ls='-')
    return 0


def figure3(x_coords, y_coords, goal_x, goal_y, x_path, y_path, w, z): # to generate Figure 3 from Foster et al. (2000)
    fig = plt.figure(3)
    plt.subplot(2, 3, 1)
    plot_Cp(w[1,:], x_coords, y_coords, fig)

    plt.subplot(2, 3, 2)
    plot_Cp(w[6, :], x_coords, y_coords, fig)

    plt.subplot(2, 3, 3)
    plot_Cp(w[21, :], x_coords, y_coords, fig)

    plt.subplot(2, 6, 1)
    plot_actions(x_coords, y_coords, goal_x, goal_y, z)

    plt.subplot(2, 6, 2)
    plot_maze(x_coords, y_coords, goal_x, goal_y, x_path[1, :], y_path[1, :])

    plt.subplot(2, 6, 3)
    plot_actions(x_coords, y_coords, goal_x, goal_y, z)

    plt.subplot(2, 6, 4)
    plot_maze(x_coords, y_coords, goal_x, goal_y, x_path[6, :], y_path[6, :])

    plt.subplot(2, 6, 5)
    plot_actions(x_coords, y_coords, goal_x, goal_y, z)

    plt.subplot(2, 6, 6)
    plot_maze(x_coords, y_coords, goal_x, goal_y, x_path[21, :], y_path[21, :])

    plt.show()


