import numpy as np
import matplotlib.pyplot as plt
import math


# place cell i = 1, ..., N as a Gaussian of the location p:
def place_cells(p, N=493, sigma=0.16):
    # place fields distributed throughout the maze:
    s = np.zeros((N, 2))
    phi = (np.sqrt(5.0) + 1.0) * 0.5
    boundary_points = round(1.0 * np.sqrt(N))
    for n in range(1, N + 1):
        if n > (N - boundary_points):
            r = 1.0
        else:
            r = np.sqrt(n - 1.0/2.0) / np.sqrt(N - (boundary_points + 1.0)/2.0)
        theta = 2.0 * np.pi * float(n)/np.power(phi, 2)
        s[n-1, 0] = r * np.cos(theta)
        s[n-1, 1] = r * np.sin(theta)

    # place cell activity is Gaussian shaped:
    f = [np.exp(-(np.linalg.norm([p, s[i, :]], ord=2)/(2.0 * np.power(sigma, 2.0)))) for i in range(N)]
    return f


# the critic of location p:
def critic(w, f_p):
    return np.dot(w, f_p)