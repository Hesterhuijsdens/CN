import numpy as np
np.random.seed(0)


# create agent
class Player(object):

    def __init__(self, env, a, b, beta, k):
        self.env = env
        self.pulls = [0] * k
        self.reward = [0] * k
        self.reward_list = []
        self.cumreward = [0] * k
        self.cumreward_list = [[0], [0]]
        self.bandit_list = []
        prior = (a / (a+b))**(a-1) * (1 - (a / (a+b)))**(b-1)
        self.rho = [prior, prior]
        self.rho_list = [[prior], [prior]]
        self.beta = beta

    def pick_slot_machine(self):
        bandit = np.argmax(self.rho)
        p = (np.exp(self.beta * self.rho[bandit]) / (np.exp(self.beta * self.rho[0]) + (np.exp(self.beta * self.rho[1]))))
        if self.pulls != 0 and np.random.rand() < p:
            return self.exploit()
        else:
            return self.explore()

    def exploit(self):
        bandit = np.argmax(self.rho)
        self.pulls[bandit] += 1
        return bandit

    def explore(self):
        bandit = np.random.choice(self.env.n_action)
        self.pulls[bandit] += 1
        return bandit

    def learn(self, bandit, reward):
        self.reward[bandit] += reward
        self.reward_list.append(reward)
        self.cumreward[bandit] += reward
        self.bandit_list.append(bandit)
        self.rho[bandit] = (self.reward[bandit] + 1.0) / (self.pulls[bandit] + 2.0)
        for machine in range(0, 2):
            self.cumreward_list[machine].append(self.cumreward[machine])
            self.rho_list[machine].append(self.rho[machine])



