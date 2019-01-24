import numpy as np
from equations_BM import *
from binary_data import *
import pickle
import pylab
import math


def poisson(lam, k):
    return np.divide(float(np.power(lam, k)), math.factorial(k)) * np.exp(-1. * lam)

# predictions (#spikes, prob)
ycount = np.zeros(11)
states = list(product([-1, 1], repeat=10))
for state in states:
    p = poisson(sum(np.array(state) == 1), 0)
    ycount[sum(np.array(state) == 1)] += p
    # plt.scatter(sum(np.array(state) == 1), p)

# data (spike train > # neurons)
bint = np.loadtxt('bint.txt')
bint = bint[20:30, :100]
bint[bint < 0.5] = -1

# observations (count occurrence)
N = np.shape(bint)[1]
xcount = np.zeros(11)
for k in range(N):
    for j in states:
        if (np.array(bint[:, k]) == np.array(j)).all():
            xcount[sum(np.array(j) == 1)] += 1
print xcount

plt.figure()

# 1024 dots in plot
for state in states:

    # what is observed
    x = xcount[sum(np.array(state) == 1)]

    # what is predicted
    y = ycount[sum(np.array(state) == 1)]
    plt.scatter(x, y)
# plt.xscale('log')
# plt.yscale('log')
plt.show()

# for observed
# rate_obs = np.asarray(observed_pattern) / (np.shape(s)[0])
# Z = np.sum(np.exp(E_list))
# p_s = ((1 / Z) * np.exp(E_list))


# n = 200
# w, b, weightlist, wsum, bsum = boltzmann_train(bint, eta=0.01, n_epochs=n)



# print ycount, sum(ycount)
#
# with open('outfile.dat', 'wb') as f:
#     pickle.dump([w, b, weightlist, wsum, bsum, bint, xcount, ycount, states], f)

# with open('outfile.dat', 'r') as f:
#     w, b, weightlist, wsum, bsum, bint, xcount, ycount, states = pickle.load(f)

# # # shape: (160L,  283041L) (spike train altijd langer dan #neuronen)
# bint = np.loadtxt('bint_small.txt')
# bint = bint[20:30, :]
# bint[bint < 0.5] = -1
#
# print np.shape(bint)
# xcount = np.zeros(11)
# N = np.shape(bint)[1]
# for k in range(N):
#     for j in states:
#         if (np.array(bint[:, k]) == np.array(j)).all():
#             xcount[sum(np.array(j) == 1)] += 1






# print poisson(2, 10)

# make 2^10 = 1024 combinations

# plt.figure()
# for j in states:
#     plt.scatter(sum(np.array(j) == 1), poisson(sum(np.array(j) == 1), 0))
# plt.yscale('log')




# counter = np.zeros(11)
# for k in range(np.shape(bint)[1]):
#
#     # count how many times it occurs
#     counter[sum(np.array(bint[:, k]) == 1)] += 1
#
#     # plt.scatter(sum(np.array(bint[:, k]) == 1), )
# # print counter
# plt.scatter(range(0, 10), counter)
# plt.yscale('log')

# plt.show()




# # test BM
# X_sample = boltzmann_dream(w, b)
# plt.figure()
# plt.imshow(X_sample)
#
# plt.figure()
# for i in range(0, w.shape[0]):
#     for j in range(0, w.shape[0]):
#         plt.plot(range(0, n), weightlist[:, i, j], label=(i, j))
# plt.xlabel('iterations')
# plt.ylabel('change in weights')
# plt.title('Convergence of change in weights')
#
# plt.figure()
# plt.plot(range(0, n), wsum)
# plt.xlabel('iterations')
# plt.ylabel('change in sum of weights')
# plt.title('Convergence of change in summed weights')
# plt.show()


