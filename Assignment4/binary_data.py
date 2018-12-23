import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0)


# generate matrix with random patterns
def get_random_pattern(N, lb):
    return np.random.choice((-1., 1.), size=(N, lb))


# plt.figure()
# for k in range(0, 6):
#     plt.subplot(3, 2, k+1)
#     plt.imshow(random_patterns[k])
# plt.show()


# generate random w
def get_w(N):
    w = np.random.uniform(-1., 1., size=(N, N))
    np.fill_diagonal(w, 0)
    return w


# generate random b
def get_b(N):
    return np.random.choice((-1., 1.), size=N)


def get_bint(patterns):
    bint = np.loadtxt('bint.txt')
    return bint[:, :patterns]






