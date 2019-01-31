import matplotlib.pyplot as plt
from environment import *
from agent import *
import seaborn as sns
import math
np.random.seed(0)


# create two-armed bandit environment
k = 2
env = KArmedBandit(k=k)
n_iter = 1000

fig1 = plt.figure()
fig2 = plt.figure()
betas = [0.01, 0.1, 1, 10]
leg = ["slot A", "slot B", "Total"]
for i in range(len(betas)):

    # create agent that plays machines
    agent = Player(env, a=1, b=1, beta=betas[i], k=k)

    # step: obs
    for pull in range(n_iter):
        action = agent.pick_slot_machine()
        reward = env.play(action)
        agent.learn(action, reward)

    total = [sum(x) for x in zip(agent.cumreward_list[0], agent.cumreward_list[1])]

    # plot rho
    ax1 = fig1.add_subplot(2, 2, i + 1)
    fig1.subplots_adjust(wspace=0.9, hspace=0.8)
    fig1.suptitle("State beliefs, A=%1.1f, B=%1.1f" % (agent.env.p[0], agent.env.p[1]))
    for belief in range(0, 2):
        ax1.plot(np.linspace(0, n_iter + 1, n_iter + 1), agent.rho_list[belief], linestyle="-", label=chr(belief + 65))
        ax1.axhline(y=agent.env.p[belief], linestyle=':')
    ax1.set_title("beta=%1.2f, total=%1.1f" % (betas[i], total[n_iter]))
    ax1.set_xlabel("time")
    ax1.set_ylabel("belief")
    ax1.legend()

    # plot cum sum
    ax2 = fig2.add_subplot(2, 2, i + 1)
    fig2.subplots_adjust(wspace=0.9, hspace=0.8)
    fig2.suptitle("Cumulative reward, A=%1.1f, B=%1.1f" % (agent.env.p[0], agent.env.p[1]))
    for machine in range(0, 2):
        ax2.plot(np.linspace(0, n_iter+1, n_iter+1), agent.cumreward_list[machine], linestyle="-")
    ax2.plot(np.linspace(0, n_iter+1, n_iter+1), total, linestyle=":", color="grey")
    ax2.set_title("beta=%1.2f, total=%1.1f" % (betas[i], total[n_iter]))
    ax2.legend(leg)
    ax2.set_xlabel("time")
    ax2.set_ylabel("cumulative reward")
    print "Beta: ", betas[i]
    print "pick slot A: ", sum(np.array(agent.bandit_list) == 0)
    print "pick slot B: ", sum(np.array(agent.bandit_list) == 1)
plt.show()




