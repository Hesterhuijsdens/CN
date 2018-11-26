import numpy as np
import matplotlib.image as img
import matplotlib.pyplot as plt
from generatePatches import *

patches = generate_data("natural")

# visualise patches:
fig = plt.figure(0)
for plot_nr in range(1, 65):
    ax = fig.add_subplot(8, 8, plot_nr)
    ax.imshow(np.reshape(patches[plot_nr-1, :], (8, 8)), cmap=plt.get_cmap(name='gray'))
    plt.axis('off')

m = np.mean(patches, axis=0)
patches = patches - np.ones((np.shape(patches)[0], 1)) * m

w = np.random.normal(size=(64, 64))
max_iteration = 100
learning_rate = 0.1

#for epoch in range(max_iteration):





# visualize PCAs:
#fig = plt.figure(1)
# for plot_nr in range(1, 65):
#     ax = fig.add_subplot(8, 8, plot_nr)
#     ax.imshow(patches[plot_nr-1, :, :], cmap=plt.get_cmap(name='gray'))
#     plt.axis('off')


plt.show()



