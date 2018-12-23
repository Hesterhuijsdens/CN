import numpy as np
import random


# two-armed bandit problem: play the machine and update belief
class KArmedBandit(object):
    # certainty agent
    def __init__(self, k):
        self.n_action = k
        self.p = [0.8, 0.2]
        # self.pick_bandit = None

    # execute action 0 / 1
    def play(self, action):
        """ return reward"""
        if random.random() < self.p[action]:
            return 1
        else:
            return -1


