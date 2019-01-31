import numpy as np
import matplotlib.image as img
import matplotlib.pyplot as plt
from generatePatches import *
from equations_sparse_coding import *
import pickle

np.random.seed(40)
#
# # load data and standardize:
# x = generate_data("pixels")
# m = np.mean(x, axis=0)
# #x = x - np.ones((np.shape(x)[0], 1)) * m
# x = x / np.linalg.norm(x, ord=2)
#
# # eigenvalue decomposition:
# eigenvalues, eigvectors = np.linalg.eig(np.cov(np.transpose(x)))
# eigvectors = np.transpose(eigvectors)
#
#
# # SPARSE CODING:
# phi = eigvectors
# a = np.trapz(np.matmul(x, phi), axis=0)
# norm = np.linalg.norm(a, ord=2)
# a_pca = a
# a = a / norm
#
# # store data dimensions etc:
# p = np.shape(x)[0]
# n = np.shape(x)[1]
# decay = 0.1
# max_iterations = 100
# sigma = 0.4
#
# converged = False
# epoch = 0
#
# while not converged or epoch < 10:
#     print "epoch", epoch
#     converged = True
#     for image in range(p):
#         for i in range(1, n): # for every dimension
#             b = b_i(phi[i], x[image])
#             C = C_ij(phi[i], phi)
#             dA = b - np.matmul(C, a) - (decay / sigma) * derivative_S(a[i], sigma)
#             a[i] = a[i] + 0.9 * dA
#             if np.abs(dA) >= 0.2:
#                 converged = False
#     epoch += 1
#
# print "number of iterations: ", epoch
# print a
#
# with open('outfile.dat', 'wb') as f:
#     pickle.dump([a, epoch, a_pca, p, norm], f)

with open('outfile.dat', 'r') as f:
    a, epoch, a_pca, p, norm = pickle.load(f)


# average over coefficients:
a = a * norm
a = (a - np.mean(a)) /np.std(a)

nr_of_bins = (np.max(a) - np.min(a)) / 0.04
bin_counters = np.zeros(int(nr_of_bins))
for bin in range(int(nr_of_bins)):
    for a_i in a:
        if a_i >= bin * 0.14 or a_i <= (bin+1) * 0.04:
            bin_counters[bin] += 1

P = bin_counters /64
bin_counters = -1.0 * np.log(P)

# do the same for PCA:
a_pca = (a_pca - np.mean(a_pca)) /np.std(a_pca)
nr_of_bins_pca = (np.max(a_pca) - np.min(a_pca)) / 0.04

print nr_of_bins_pca
bin_counters_pca = np.zeros(int(nr_of_bins_pca))
for bin in range(int(nr_of_bins_pca)):
    for a_i in a_pca:
        if a_i >= bin * 0.14 or a_i <= (bin + 1) * 0.04:
            bin_counters_pca[bin] += 1

P = bin_counters_pca / 64
bin_counters_pca = -1.0 * np.log(P)

plt.figure(2)
plt.plot(np.linspace(np.min(a) + 4, np.max(a) + 4, int(nr_of_bins)), bin_counters, label='sparse', color='green')
plt.plot(np.linspace(np.min(a_pca) +1, np.max(a_pca) + 1, int(nr_of_bins_pca)), bin_counters_pca, label='pca', color='blue')
plt.xlabel('a_i')
plt.ylabel('p(a_j)')
plt.title('Frequencies of a using pixels')
plt.legend()
plt.show()