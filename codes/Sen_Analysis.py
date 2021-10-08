# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import math
import random
import matplotlib.pyplot as plt
import pandas as pd

from utilities import randomInt
from exp3M import exp3m
from ucb import ucb_bandit
from eps_greedy import eps_bandit

plot_dir = r"E:\Codes\adversarial_multi_armed_bandit_variable_plays\results\plots"

# Initialization
N = 7000
d = 10

# Importing the dataset
reward_vector = pd.read_csv("car_hacking_dataset.csv")[1600:]


# Implementing Exp3M.VP
numPlays_std = 0.8
gamma_exp3m_vp = 0.07

bounds_pool = (
    # (1, 3),
    # (2, 4),
    (3, 5),
    (4, 6),
    (5, 7),
    (6, 8),
    # (7, 9),
)

cumulative_reward_exp3m_vp = np.zeros((len(bounds_pool), N))

for i, (numPlays_LB, numPlays_UB) in enumerate(bounds_pool):
    numPlays_exp3m_vp = randomInt(numPlays_LB, numPlays_UB, numPlays_std, N)
    mean_reward_exp3m_vp = 0
    t = 0

    for (choice, reward, est, weights) in exp3m(
        d,
        lambda t, choice: [reward_vector.values[t, i] for i in choice],
        gamma_exp3m_vp,
        numPlays_exp3m_vp,
    ):
        mean_reward_exp3m_vp = mean_reward_exp3m_vp + (
            reward - mean_reward_exp3m_vp
        ) / (t + 1)
        cumulative_reward_exp3m_vp[i, t] = mean_reward_exp3m_vp

        t += 1

        if t >= N:
            break

# Implementing Exp3.M
numPlay_pool = range(4, 8)
gamma_exp3m = 0.07
cumulative_reward_exp3m = np.zeros((len(numPlay_pool), N))

for i, n in enumerate(numPlay_pool):
    numPlays_exp3m = [n] * N
    mean_reward_exp3m = 0
    t = 0

    for (choice, reward, est, weights) in exp3m(
        d,
        lambda t, choice: [reward_vector.values[t, i] for i in choice],
        gamma_exp3m,
        numPlays_exp3m,
    ):
        mean_reward_exp3m = mean_reward_exp3m + (reward - mean_reward_exp3m) / (t + 1)
        cumulative_reward_exp3m[i, t] = mean_reward_exp3m

        t += 1

        if t >= N:
            break

# Plotting
fig, axs = plt.subplots(nrows=2, ncols=2, constrained_layout=True)
for i, ax in enumerate(a for sublist in axs for a in sublist):
    ax.plot(cumulative_reward_exp3m[i, :], label="Exp3.M, M={}".format(numPlay_pool[i]))
    ax.plot(
        cumulative_reward_exp3m_vp[i, :],
        label="Exp3.M-VP, \u03BD={}".format(int(np.mean(bounds_pool[i]))),
    )
    ax.set_xlim(left=0, right=N)
    # ax.set_ylim(top=1)
    ax.legend(loc="best")
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Average Reward")
    # ax.set_title("Average Rewards")
    ax.grid(True)

fig.savefig(plot_dir + "\\compare_M_{}_to_{}.png".format(4, 7))
