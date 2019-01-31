from binary_data import *
from itertools import *


def E(x, w, b):
    return 0.5 * (np.dot(np.dot(w, x), x) + np.dot(b, x))


def z(X, w, b):
    n, p = np.shape(X)

    # all possible states
    states = list(product([-1, 1], repeat=n))
    Z = 0
    for state in states:
        Z += np.exp(-E(state, w, b))
    return Z


def state_prob(X, w, b, Z):
    return 1./Z * (np.exp(E(X, w, b)))


def flip_prob(w, x, b, j):
    part1 = 0.5 * np.dot(w, x) + b[j]
    part2 = part1 * x[j]
    return 1. / (1. + np.exp(part2))


def gibbs_sampling(w, b, n_gibbs=500, n_burnin=10):
    n_nodes = w.shape[0]
    X = np.zeros((n_nodes, n_gibbs))
    X[:, 0] = np.random.randint(2, size=n_nodes)
    for i in range(n_nodes):
        if X[i, 0] >= 0.5:
            X[i, 0] = 1.0
        else:
            X[i, 0] = -1.0

    for i in range(1, n_gibbs):
        for j in range(n_nodes):
            p = flip_prob(w[:, j], X[:, i-1], b, j)
            if (np.random.rand() < p).astype("float"):
                X[j, i] = -X[j, i-1]
            else:
                X[j, i] = X[j, i-1]

    # discard burn-in (depend on state initialisation)
    return X[:, n_burnin:]


# Compute expectations
def stats(X):
    dw = np.cov(X)
    np.fill_diagonal(dw, 0)
    db = np.mean(X, axis=1)
    return dw, db


def log_likelihood(X, w, b, Z):
    n, p = np.shape(X)
    ps = np.zeros((p, n))
    for pi in range(p):
        ps[pi] = state_prob(X[:, pi], w, b, Z)
    return np.mean(np.sum(np.log(ps), axis=1))


def boltzmann_train(patterns, eta, n_epochs=200, n_gibbs=500, n_burnin=10):
    n_nodes, n_examples = np.shape(patterns)
    w = get_w(n_nodes)
    w_list = np.zeros((n_epochs, n_nodes, n_nodes))
    w_sum = np.zeros(n_epochs)
    b_sum = np.zeros(n_epochs)
    b = np.zeros(n_nodes)
    Z = z(patterns, w, b)
    print log_likelihood(patterns, w, b, Z)

    # clamped statistics
    dw_c, db_c = stats(patterns)

    for i_epoch in range(n_epochs):

        # free statistics
        XM = gibbs_sampling(w, b, n_gibbs, n_burnin)
        dw_free, db_free = stats(XM)

        w += (eta * (dw_c - dw_free))
        b += (eta * (db_c - db_free))

        # w_list[i_epoch, :, :] = w
        w_list[i_epoch, :, :] = eta * (dw_c - dw_free)

        w_sum[i_epoch] = np.sum(np.abs((eta * (dw_c - dw_free))))
        b_sum[i_epoch] = np.sum(np.abs((eta * (db_c - db_free))))

        # force symmetry
        w = (w + w.T) / 2

    print log_likelihood(patterns, w, b, Z)
    return w, b, w_list, w_sum, b_sum


# Boltzmann dreaming
def boltzmann_dream(w, b, n_epochs=20):
    return gibbs_sampling(w, b, n_gibbs=20, n_burnin=10)
