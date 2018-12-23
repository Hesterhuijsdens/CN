import matplotlib.pyplot as plt
from environment import *
from agent import *
import math

# create two-armed bandit environment
k = 2
env = KArmedBandit(k=k)
n_iter = 100

plt.figure()
betas = [0.01, 0.1, 1, 10]
for i in range(len(betas)):

    # create agent that plays machines
    agent = Player(env, a=1, b=1, beta=betas[i], k=k)

    # step: obs
    for pull in range(n_iter):
        action = agent.pick_slot_machine()
        reward = env.play(action)
        agent.learn(action, reward)

    plt.subplot(2, 2, i+1)
    leg = ["slot A", "slot B"]
    for j in range(n_iter):
        if agent.bandit_list[j] == 0:
            plt.scatter(j, agent.reward_list[j], c="blue")
        else:
            plt.scatter(j, agent.reward_list[j], c="red")
    plt.title("2-Armed-Bandit with beta=%f1.2" % betas[i])
    plt.legend(leg)
plt.show()




