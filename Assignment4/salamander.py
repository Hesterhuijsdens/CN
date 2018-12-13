import numpy as np
from equations_BM import *
from binary_data import *


# shape: (160L, 283041L) (spike train altijd langer dan #neuronen)
bint = np.loadtxt('bint_small.txt')
print np.shape(bint)

n = 200
w, b, weightlist = boltzmann_train(bint, eta=0.001, n_epochs=n)

# test BM
X_sample = boltzmann_dream(w, b)
plt.figure()
plt.imshow(X_sample)

plt.figure()
for i in range(0, w.shape[0]):
    for j in range(0, w.shape[0]):
        plt.plot(range(0, n), weightlist[:, i, j], label=(i, j))
plt.show()


