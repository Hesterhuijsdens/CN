import numpy as np


# computes cost with sparsity penalty:
def cost_sparseness(I, a, phi, lambda_):
    total_cost = 0
    for image in I:
        total_cost += np.sum(np.power(image - np.matmul(a, phi), 2))
    return (1.0/np.shape(I)[0]) * total_cost


def b_i(phi, image, i):
    return np.dot(phi[i, :], image)


def C_ij(phi, i, j):
    return np.dot(phi[i, :], phi[j, :])
