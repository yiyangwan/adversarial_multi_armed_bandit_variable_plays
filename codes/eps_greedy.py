# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import random


# Implementing Random Selection
class eps_bandit:
    """
    epsilon-greedy k-bandit problem
    
    Inputs
    =====================================================
    k: number of arms (int)
    eps: probability of random action 0 < eps < 1 (float)
    iters: number of steps (int)
    df: reward vector for each arm and at each time -- dataframe
    """

    def __init__(self, k, eps, iters, df):
        # Number of arms
        self.k = k
        # Search probability
        self.eps = eps
        # Number of iterations
        self.iters = iters
        # Step count
        self.n = 0
        # Step count for each arm
        self.k_n = np.zeros(k)
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
        # Generate random number
        p = np.random.rand()
        if self.eps == 0 and self.n == 0:
            a = np.random.choice(self.k)
        elif p < self.eps:
            # Randomly select an action
            a = np.random.choice(self.k)
        else:
            # Take greedy action
            a = np.argmax(self.k_reward)

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

    def reset(self):
        # Resets results while keeping settings
        self.n = 0
        self.k_n = np.zeros(self.k)
        self.mean_reward = 0
        self.reward = np.zeros(self.iters)
        self.k_reward = np.zeros(self.k)
        self.a = 0
        self.current_reward = 0

    def run(self):
        for i in range(self.iters):
            self.pull()
            self.reward[i] = self.mean_reward


def eps_greedy(k, iters, epsilon, dataset):
    """
    input:  d -- number of arms
            N -- number of trails
            dataset -- reward vector
    """

    eps = eps_bandit(k, epsilon, iters, dataset)

    for n in range(0, eps.iters):
        eps.pull()
        eps.reward[n] = eps.mean_reward
        yield eps.a, eps.current_reward, eps.mean_reward * eps.n


# # Implementing UCB
# def ucb(d, N, dataset):
#     """
#     input:  d -- number of arms
#             N -- number of trails
#             dataset -- reward vector
#     """
#     ads_selected = []
#     numbers_of_selections = [0] * d
#     sums_of_reward = [0] * d
#     total_reward = 0

#     for n in range(0, N):
#         ad = 0
#         max_upper_bound = 0
#         for i in range(0, d):
#             if numbers_of_selections[i] > 0:
#                 average_reward = sums_of_reward[i] / numbers_of_selections[i]
#                 delta_i = math.sqrt(2 * math.log(n + 1) / numbers_of_selections[i])
#                 upper_bound = average_reward + delta_i
#             else:
#                 upper_bound = 1e400
#             if upper_bound > max_upper_bound:
#                 max_upper_bound = upper_bound
#                 ad = i
#         ads_selected.append(ad)
#         numbers_of_selections[ad] += 1
#         reward = dataset.values[n, ad]
#         sums_of_reward[ad] += reward
#         total_reward += reward
#         yield ad, reward, total_reward


if __name__ == "__main__":

    N = 100000
    d = 26
    # Importing the dataset
    dataset = pd.read_csv("car_hacking_dataset.csv")

    for ad, reward, total_reward in eps_greedy(d, N, 0.1, dataset):
        print("ad:%d\treward:%d" % (ad, total_reward))
