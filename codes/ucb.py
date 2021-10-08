# import modules
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class ucb_bandit:
    """
    Upper Confidence Bound Bandit
    
    Inputs 
    ============================================
    k: number of arms (int)
    c:
    iters: number of steps (int)
    df: reward vector for each arm and at each time -- dataframe
    """

    def __init__(self, k, c, iters, df):
        # Number of arms
        self.k = k
        # Exploration parameter
        self.c = c
        # Number of iterations
        self.iters = iters
        # Step count
        self.n = 1
        # Step count for each arm
        self.k_n = np.ones(k)
        # Total mean reward
        self.mean_reward = 0
        self.reward = np.zeros(iters)
        # Mean reward for each arm
        self.k_reward = np.zeros(k)
        # Reward on each trail
        self.current_reward = 0
        # Reward vector for each arm on each trail
        self.reward_vector = df
        # Choice on each trail
        self.a = 0

    def pull(self):
        # Select action according to UCB Criteria
        a = np.argmax(self.k_reward + self.c * np.sqrt((np.log(self.n)) / self.k_n))

        reward = self.reward_vector.values[self.n, a]
        self.current_reward = reward
        # Update counts
        self.n += 1
        self.k_n[a] += 1

        # Update total
        self.mean_reward = self.mean_reward + (reward - self.mean_reward) / self.n

        # Update results for a_k
        self.k_reward[a] = self.k_reward[a] + (reward - self.k_reward[a]) / self.k_n[a]

        self.a = a

    def run(self):
        for i in range(self.iters):
            self.pull()
            self.reward[i] = self.mean_reward

    def reset(self, mu=None):
        # Resets results while keeping settings
        self.n = 1
        self.k_n = np.ones(self.k)
        self.mean_reward = 0
        self.reward = np.zeros(self.iters)
        self.k_reward = np.zeros(self.k)
        self.a = 0
        self.current_reward = 0


def ucb(d, c, N, dataset):
    """
    input:  d -- number of arms
            N -- number of trails
            c -- exploration parameter
            dataset -- reward vector
    """
    ucb_algorithm = ucb_bandit(d, c, N, dataset)

    for n in range(0, ucb_algorithm.iters):
        ucb_algorithm.pull()
        ucb_algorithm.reward[n] = ucb_algorithm.mean_reward
        yield ucb_algorithm.a, ucb_algorithm.current_reward, ucb_algorithm.mean_reward * ucb_algorithm.n


if __name__ == "__main__":
    c = 2
    N = 100000
    d = 26
    # Importing the dataset
    dataset = pd.read_csv("car_hacking_dataset.csv")

    for ad, reward, total_reward in ucb(d, c, N, dataset):
        print("ad:%d\treward:%d" % (ad, total_reward))
