import numpy as np
import matplotlib.image as img
import matplotlib.pyplot as plt
from generatePatches import *
from equations_sparse_coding2 import *

# load data and standardize:
x = generate_data("natural")
m = np.mean(x, axis=0)
x = x - np.ones((np.shape(x)[0], 1)) * m

# eigenvalue decomposition:
a, phi = np.linalg.eig(np.cov(np.transpose(x))) # eigenvalues, eigenvectors
phi = np.transpose(phi)

# visualize PCA components:
fig = plt.figure(0)
for component in range(1, np.shape(phi)[0] + 1):
    ax = fig.add_subplot(8, 8, component)
    ax.imshow(np.reshape(phi[component-1, :], (8, 8)), cmap=plt.get_cmap(name='gray'))
    plt.axis('off')
plt.show()

# set learning parameters and initialize:
lambda_ = 0.0           # sparsity penalty
eta = 0.03              # learning rate
max_iteration = 10

# create array to store costs:
cost_sparse = np.zeros(max_iteration)

# train values for a:
for epoch in range(max_iteration):

    cost_sparse[epoch] = cost_sparseness(x, a, phi, lambda_)
    for image in x:
        # for every component (coordinate descent?):
        for i in range(np.shape(a)[0]):
            a_new = b_i(phi, image, i)

            print "anew: ", a_new

            # update coefficients a:
            a = (1.0 - eta) * a + eta * a_new

