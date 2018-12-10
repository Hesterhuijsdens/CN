import numpy as np
from PreprocessData import load37
from binary_data import *
from equations_BM import *


# avoid overflow warnings
np.seterr(all="ignore")


# for now, 3 patterns on a network of 5 neurons!
patterns = np.loadtxt('data.txt')
w, b = boltzmann_train(patterns, n_epochs=200)

# test BM
# X_sample = boltzmann_dream(w, b)
# plt.figure()
# plt.imshow(X_sample)
# plt.show()


# TO DO:
# plot change in weights vs iteration
# mean field theory and linear regression correction
# load in MNIST data
# build classifier (2.5.1)









