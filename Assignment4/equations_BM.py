import numpy as np
import matplotlib.pyplot as plt
from itertools import *


def sigmoid(A):
    return 1 / (1 + np.exp(-A))


def flip_prob(w, x, b):
    """ probability """
    return sigmoid(np.dot(w, x) + b)


def E(x, w, b):
    """ compute energy for a particular state """
    n = len(x)
    E = 0
    for i in range(n):
        for j in range(n):
            if i is not j:
                E +=0.5 * w[i, j] * x[i] * x[j] + b[i] * x[i]
    return E


def z(X, w, b):
    n, p = X.shape

    # all possible neuronal states: 2^N = 2^4 = 16
    states = list(product([-1, 1], repeat=n))

    Z = 0
    for state in states:
        Z += np.exp(-E(state, w, b))
    return Z


def state_prob(X, w, b, pi):
    """ states of low energy should have high probs """
    n, p = X.shape
    Z = z(X, w, b)
    return 1./Z * (np.exp(-E(X[:, pi], w, b)))


def gibbs_sampling(w, b, n_gibbs, n_burnin):
    """ approximate model distribution for training a BM """
    # 10 nodes
    n_nodes = w.shape[0]

    # array for saving node states for each time step
    X = np.zeros((n_nodes, n_gibbs))

    # state vector initialisation (t=0)
    X[:, 0] = np.random.randint(2, size=n_nodes)

    # loop over Gibbs samples
    for i in range(1, n_gibbs):

        # loop over nodes
        for j in range(n_nodes):

            # compute flip probability
            p = flip_prob(w[:, j], X[:, i-1], b[j])

            # determine new binary state
            if (np.random.rand() < p).astype("float"):
                X[j, i] = 1.
            else:
                X[j, i] = -1.

    # discard burn-in (depend on state initialisation)
    return X[:, n_burnin:]


# Compute expectations
def compute_expectations(X):
    """ compute the expectation (mean over patterns / samples) of the
    partial derivatives for w and b"""

    # 1 pattern is training example of length m
    dw = (np.dot(X, X.T)) / X.shape[1]
    np.fill_diagonal(dw, 0)
    db = np.mean(X, axis=1)
    return dw, db


def log_likelihood(X, w, b):
    n, p = X.shape
    ps = np.zeros((p, n))

    # loop over patterns
    for pi in range(p):
        ps[pi] = state_prob(X, w, b, pi)
    return np.mean(np.sum(np.log(ps), axis=1))


def boltzmann_train(patterns, eta=0.01, n_epochs=30, n_gibbs=500, n_burnin=10):
    n_nodes, n_examples = patterns.shape

    # weights initialisation
    w = np.loadtxt('w.txt')

    # bias initialisation
    b = np.zeros(n_nodes)
    # b = np.loadtxt('b.txt')

    # E(patterns[:, 2], w, b)
    # state_prob(patterns, w, b, 2)
    log_likelihood(patterns, w, b)

    # expectations under empirical distribution (training patterns)
    dE_dw, dE_db = compute_expectations(patterns)

    # print E(3, patterns[:, 2], w, b), state_prob(patterns, w, b, 2, 1)

    # loop over epochs
    for i_epoch in range(n_epochs):
        # print("Epoch {}/{}.".format(1 + i_epoch, n_epochs))

        # Gibbs sampling with current model: free stats
        XM = gibbs_sampling(w, b, n_gibbs, n_burnin)

        # expectations under model distribution:
        dEM_dw, dEM_db = compute_expectations(XM)

        # update weights and biases
        w += (eta * (dEM_dw - dE_dw))
        b += (eta * (dEM_db - dE_db))

        # E should go down, prob should go up
        # print E(patterns[:, 0], w, b), state_prob(patterns, w, b, 0)

    # force symmetry
    w = (w + w.T) / 2

    log_likelihood(patterns, w, b)
    return w, b


# Boltzmann dreaming
def boltzmann_dream(w, b, n_epochs=20):
    return gibbs_sampling(w, b, n_gibbs=20, n_burnin=10)




