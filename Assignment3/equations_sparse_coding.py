import numpy as np
import matplotlib.pyplot as plt


# computes mean squared error between images and reconstructions:
def preserve_information(I, a, phi):
    return 0


# computes sparseness of coefficients a:
def sparsity_penalty(decay, a):
    return 0


# cost function to be minimized:
def cost_function(I, a, phi, decay):
    return -1.0 * preserve_information(I, a, phi) - sparsity_penalty(decay, a)