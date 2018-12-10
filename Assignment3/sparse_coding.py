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
plt.show()

