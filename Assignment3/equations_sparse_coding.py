import numpy as np
import matplotlib.pyplot as plt


# to compute b:
def b_i(phi, I):
    return np.dot(phi, I)


# to compute C:
def C_ij(phi, all_phi):
    return np.matmul(phi, all_phi)


# S(x) = log(1 + x^2):
def derivative_S(a, sigma):
    x = a / sigma
    return (2.0 * x) / (1.0 + np.power(x, 2))
