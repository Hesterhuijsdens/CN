import numpy as np
import matplotlib.pyplot as plt

# use random seed:
np.random.seed(5)

# initialize model parameters:
dt = 0.001          # time step
v_threhold = 1.0    # threshold for action potential
T = 20              # total time: 0 < t < T
n = 200              # nr of samples/trajectories


# ML estimates for mu and sigma:
def ml_mu(n, fpts, dt):
    fpts = fpts * dt
    return (1.0 / (np.sum(fpts) / n))


def ml_sigma(n, fpts, dt):
    fpts = fpts * dt
    return (1.0 / n) * np.sum(1.0 / fpts) - ml_mu(n, fpts, dt)


mus = [0.01, 0.1, 0.5, 1.0, 1.5]
sigmas = [0.01, 0.1, 0.5]

# initialize matrices for trajectories and FPTs:
observed_means = np.zeros((5, 3, n))
observed_stds = np.zeros((5, 3, n))
ml_means = np.zeros((5, 3, n))
ml_stds = np.zeros((5, 3, n))

index = 0
for mu in mus:
    for sigma in sigmas:
        v = np.zeros((n, int(T / dt)))
        FPT = np.full(n, T)

        # generate n trajectories with total time T:
        for i in range(n):
            for t in range(int(T / dt) - 1):
                v[i, t + 1] = v[i, t] + mu * dt + np.random.normal(loc=0.0, scale=sigma * np.sqrt(dt))
                if v[i, t + 1] > v_threhold:
                    v[i, t + 1] = 0.0

                    # store first passage time:
                    if FPT[i] == T:
                        FPT[i] = t + 1

            observed_means[index / 3, index % 3, i] = np.mean(FPT[0:(i + 1)] * dt)
            observed_stds[index / 3, index % 3, i] = np.std(FPT[0:(i + 1)] * dt)
            ml_means[index / 3, index % 3, i] = ml_mu(i + 1, FPT[0:(i + 1)], dt)
            ml_stds[index / 3, index % 3, i] = np.sqrt(ml_sigma(i + 1, FPT[0:(i + 1)], dt))
        index += 1

plt.figure()
fig, ax = plt.subplots(5, 3)
plt.rcParams['figure.figsize'] = [26, 23]
plt.subplots_adjust(hspace=0.5)
for m in range(3):
    for s in range(5):
        ax[s, m].plot(range(n), ml_means[s, m, 0:n + 1], label='ML mu')
        ax[s, m].plot(range(n), observed_means[s, m, 0:n + 1], label='model mu')
        ax[s, m].plot(range(n), ml_stds[s, m, 0:n + 1], label='ML stds')
        ax[s, m].plot(range(n), observed_stds[s, m, 0:n + 1], label='model stds')
        ax[s, m].set_title('mu = ' + str(mus[s]) + ', sigma = ' + str(sigmas[m]))
        ax[s, m].set_xlabel('n')

plt.show()
