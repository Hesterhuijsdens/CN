import numpy as np
from binary_data import *
from equations_BM import *


# avoid overflow warnings
np.seterr(all="ignore")


# for now, 3 patterns on a network of 5 neurons!
patterns = get_random_pattern(100, 10)
n = 200
w, b, weightlist = boltzmann_train(patterns, eta=0.001, n_epochs=n)

# test BM
X_sample = boltzmann_dream(w, b)
plt.figure()
plt.imshow(X_sample)

plt.figure()
for i in range(0, w.shape[0]):
    for j in range(0, w.shape[0]):
        plt.plot(range(0, n), weightlist[:, i, j], label=(i, j))

plt.figure()
plt.plot(range(0, n), weightlist[:, 3, 4])
plt.legend()
plt.show()


# TO DO:
# plot change in weights vs iteration
# mean field theory and linear regression correction
# load in MNIST data
# build classifier (2.5.1)









