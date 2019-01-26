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
leg = ["slot A", "slot B"]
for i in range(len(betas)):

    # create agent that plays machines
    agent = Player(env, a=1, b=1, beta=betas[i], k=k)

    # step: obs
    for pull in range(n_iter):
        action = agent.pick_slot_machine()
        reward = env.play(action)
        agent.learn(action, reward)

    # plt.subplot(2, 2, i+1)
    # for j in range(n_iter):
    #     if agent.bandit_list[j] == 0:
    #         plt.scatter(j, agent.reward_list[j], c="blue")
    #     else:
    #         plt.scatter(j, agent.reward_list[j], c="red")
    # plt.title("2-Armed-Bandit with beta=%1.2f" % betas[i])
    # plt.xlabel("time")
    # plt.ylabel("reward")
    # plt.subplots_adjust(wspace=0.9, hspace=0.8)
    # plt.legend(leg)

    plt.suptitle("Cumulative reward against iterations")
    for machine in range(0, 2):
        plt.subplot(2, 2, i + 1)
        plt.plot(np.linspace(0, n_iter+1, n_iter+1), agent.cumreward_list[machine])
        plt.title("beta=%1.2f" % betas[i])
    plt.legend(leg)
    plt.subplots_adjust(wspace=0.9, hspace=0.8)
plt.show()




