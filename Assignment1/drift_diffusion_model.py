import numpy as np
import matplotlib.pyplot as plt
from equations_drift_diffusion import *

# use random seed:
np.random.seed(5)

# initialize model parameters:
dt = 0.001          # time step
mu = 0.1            # mean
sigma = 0.1         # std
v_threhold = 1.0    # threshold for action potential
T = 20              # total time: 0 < t < T
n = 500              # nr of samples/trajectories

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
locks, labels = plt.xticks()
plt.xticks(locks, [-2.5, 0, 2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20, 22.5])
plt.xlabel('t')
plt.ylabel('FPT')
plt.title('Histogram of FPTs')

# transform estimated FPTs into probability density function to compare:
mean_FPT = np.mean(FPT)
std_FPT = np.std(FPT)

rho = np.zeros(int(T/dt))
estimated_rho = np.zeros(int(T/dt))
counter = 0
for t in np.linspace(dt, T, int(T/dt) - 1):

    # compute theoretical FPT distribution (rho):
    part_left = (v_threhold / (np.sqrt(2*np.pi) * sigma * np.power(t, 3.0/2.0)))
    part_right = np.exp(-np.power(v_threhold - mu * t, 2) / (2 * np.power(sigma, 2) * t))
    rho[counter + 1] = part_left * part_right

    # compute estimated FPT distribution:
    estimated_rho[counter + 1] = np.exp(-np.power(t - dt * mean_FPT, 2)/(2 * np.power(dt * std_FPT, 2))) / np.sqrt(2 * np.pi * np.power(std_FPT * dt, 2))
    counter += 1

print np.shape(estimated_rho)
print np.shape(rho)

plt.figure(1)
plt.plot(range(int(T/dt)), rho, label='FPT distribution')
plt.plot(range(int(T/dt)), estimated_rho, label='estimated FPT distribution')
plt.legend()
plt.xticks(locks, [-2.5, 0, 2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20, 22.5])
plt.xlabel('t')
plt.ylabel('probability')
plt.title('Estimated and true FPT distributions')
plt.show()
