import numpy as np
import matplotlib.image as img
import matplotlib.pyplot as plt
from generatePatches import *

# load data and standardize:
x = generate_data("natural")
m = np.mean(x, axis=0)
x = x - np.ones((np.shape(x)[0], 1)) * m

# eigenvalue decomposition:
eigenvalues, eigvectors = np.linalg.eig(np.cov(np.transpose(x)))
eigvectors = np.transpose(eigvectors)

# visualize PCA components:
fig = plt.figure(1)
for component in range(1, 65):
    ax = fig.add_subplot(8, 8, component)
    ax.imshow(np.reshape(eigvectors[component-1, :], (8, 8)), cmap=plt.get_cmap(name='gray'))
    plt.axis('off')
#plt.show()

# reconstruct image patches:
Z = np.matmul(x, eigvectors)
reconstruction = np.matmul(Z, np.transpose(eigvectors)) + m
print "reconstruct: ", np.shape(reconstruction)

# use the method from Olshausen et al. (1996) to optimize sparsity:
decay = 0.1 # lambda
a = np.random.rand(np.shape(x)[0], np.shape(x)[1])  # coefficients to train

# for epoch in range(20):
#     a = [for i in range(np.shape())]
#     a = np.trapz(np.dot(eigvectors, x[epoch, :]))
#     print(a)
