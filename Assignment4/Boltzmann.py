import numpy as np
from binary_data import *
from equations_BM import *

# avoid overflow warnings
np.seterr(all="ignore")


# create data
patterns = get_random_pattern(10, 10)
# plt.figure()
# plt.imshow(patterns)

n = 200
w, b, weightlist, wsum, bsum = boltzmann_train(patterns, eta=0.03, n_epochs=n)

plt.figure()
for i in range(0, w.shape[0]):
    for j in range(0, w.shape[0]):
        plt.plot(range(0, n), weightlist[:, i, j], label=(i, j))
plt.xlabel('iterations')
plt.ylabel('change in weights')
plt.title('Convergence of change in weights')

plt.figure()
plt.plot(range(0, n), wsum)
plt.xlabel('iterations')
plt.ylabel('change in sum of weights')
plt.title('Convergence of change in summed weights')
plt.show()









