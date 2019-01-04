import numpy as np
import matplotlib.pyplot as plt
from equations_drift_diffusion import *

# initialize model parameters:
dt = 0.001          # time step
mu = 0.1            # mean
sigma = 0.1         # std
v_threhold = 1.0    # threshold for action potential
T = 20              # total time: 0 < t < T
n = 300              # nr of samples/trajectories

# initialize matrices for trajectories and FPTs:
v = np.zeros((n, int(T/dt)))
FPT = np.full(n, T)

# generate n trajectories with total time T:
for i in range(n):
    for t in range(int(T/dt)-1):
        v[i, t + 1] = v[i, t] + mu * dt + np.random.normal(loc=0.0, scale=sigma * np.sqrt(dt))
        if v[i, t + 1] > v_threhold:
            v[i, t + 1] = 0.0

            # store first passage time:
            if FPT[i] == T:
                FPT[i] = t + 1

# generate histogram of FPTs:
plt.figure(0)
x, bins, patches = plt.hist(FPT, 80, facecolor='green', edgecolor='black', linewidth=1.0)
print np.shape(bins) # 81
print bins
plt.xlabel('t')
plt.ylabel('FPT')
plt.title('Estimated distribution of FPTs')

# transform estimated FPTs into probability density function to compare:
pdf_FPT = np.zeros(n)
for bin_nr in range(np.shape(bins)[0]):
    for i in range(n):


# compute theoretical FPT distribution (rho):
rho = np.zeros(int(T/dt))
counter = 0
for t in np.linspace(dt, T, int(T/dt) - 1):
    part_left = (v_threhold / (np.sqrt(2*np.pi) * sigma * np.power(t, 3.0/2.0)))
    part_right = np.exp(-np.power(v_threhold - mu * t, 2) / (2 * np.power(sigma, 2) * t))
    rho[counter + 1] = part_left * part_right
    counter += 1

plt.figure(1)
plt.plot(range(int(T/dt)), rho)
plt.show()