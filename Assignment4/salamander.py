import numpy as np
from equations_BM import *
from binary_data import *
import pickle
import pylab
import math
from scipy import stats


def poisson(lam, k):
    return np.divide(float(np.power(lam, k)), math.factorial(k)) * np.exp(-1. * lam)


# predictions (#spikes, prob): TRY 1 (not in results)
# ycount = np.zeros(1024)
# states = list(product([-1, 1], repeat=10))
# for state in range(len(states)):
#     p = poisson(sum(np.array(states[state]) == 1), 0)
#     ycount[state] = p

# predictions (#spikes, prob): TRY 2 (in results)
ycount = np.zeros(1024)
states = list(product([-1, 1], repeat=10))
for state in range(len(states)):
    prob = 10
    for i in range(len(states[0])):
        if states[state][i] == 1:
            ycount[state] *= float(1./prob)
            prob += 1

# data (spike train > # neurons)
bint = np.loadtxt('bint.txt')
bint = bint[20:30, :]
bint[bint < 0.5] = -1


# observations (count occurrence)
N = np.shape(bint)[1]
xcount = np.zeros(1024)
for obs in range(N):
    for state in range(np.shape(states)[0]):
        if (np.array(bint[:, obs]) == np.array(states[state])).all():
            xcount[state] += 1

# Boltzmann predictions
n = 200
w, b, weightlist, wsum, bsum = boltzmann_train(bint, eta=0.01, n_epochs=n)

# predictions (#spikes, prob)
zcount = np.zeros(1024)
states = list(product([-1, 1], repeat=10))
Z = np.sum(np.exp(states))
for state in range(len(states)):
    p = state_prob(states[state], w, b, Z)
    zcount[state] = p

with open('outfile.dat', 'wb') as f:
    pickle.dump([w, b, weightlist, wsum, bsum, bint, xcount, ycount, zcount, states], f)

with open('outfile.dat', 'r') as f:
    w, b, weightlist, wsum, bsum, bint, xcount, ycount, zcount, states = pickle.load(f)


plt.figure()
for i in range(0, w.shape[0]):
    for j in range(0, w.shape[0]):
        plt.plot(range(0, n), weightlist[:, i, j], label=(i, j))
plt.xlabel('iterations')
plt.ylabel('change in weights')
plt.title('Convergence of change in weights')

plt.figure()
plt.plot(range(0, n), (wsum))
plt.xlabel('iterations')
plt.ylabel('change in sum of weights')
plt.title('Convergence of change in summed weights')
plt.show()

# plot bint
ims = bint[:, :10]
plt.figure()
plt.imshow(ims)
plt.show()


# plot figure from paper
plt.figure()
leg = ["P1", "P2"]
for i in range(len(np.array(xcount))):
    plt.scatter(xcount[i] / 283041., ycount[i], c="lightgrey", s=8)
    plt.scatter(xcount[i] / 283041., zcount[i], c="red", s=8)
plt.xscale('log')
plt.yscale('log')
plt.ylim(1.e-10, 1.e2)
plt.xlim(1.e-6, 1.e2)
plt.legend(leg)
plt.xlabel("Observed pattern rate (s^-1)")
plt.ylabel("Approximated pattern rate (s^-1)")
plt.tight_layout()
plt.show()

