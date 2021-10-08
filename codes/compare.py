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
d = 6
# Importing the dataset
reward_vector = pd.read_csv("car_hacking_dataset.csv")[1600:]

# Placeholder of eps greedy
epsilon = 0.5  # exploration parameter of eps greedy
eps = eps_bandit(d, epsilon, N, reward_vector)

# Placeholder of UCB
c = 2  # exploration parameter of ucb
ucb_a = ucb_bandit(d, c, N, reward_vector)


# Implementing Exp3M.VP
numPlays_LB = 1
numPlays_UB = 3
numPlays_std = 0.1
gamma_exp3m_vp = 0.07

numPlays_exp3m_vp = randomInt(numPlays_LB, numPlays_UB, numPlays_std, N)
# numPlays_exp3m_vp = [2] * N

mean_reward_exp3m_vp = 0
cumulative_reward_exp3m_vp = np.zeros(N)

t = 0


for (choice, reward, est, weights) in exp3m(
    d,
    lambda t, choice: [reward_vector.values[t, i] for i in choice],
    gamma_exp3m_vp,
    numPlays_exp3m_vp,
):
    mean_reward_exp3m_vp = mean_reward_exp3m_vp + (reward - mean_reward_exp3m_vp) / (
        t + 1
    )
    cumulative_reward_exp3m_vp[t] = mean_reward_exp3m_vp

    t += 1

    if t >= N:
        break

# Implementing Exp3.M
numPlays_exp3m = [2] * N
gamma_exp3m = 0.07

mean_reward_exp3m = 0
cumulative_reward_exp3m = np.zeros(N)

t = 0

for (choice, reward, est, weights) in exp3m(
    d,
    lambda t, choice: [reward_vector.values[t, i] for i in choice],
    gamma_exp3m,
    numPlays_exp3m,
):
    mean_reward_exp3m = mean_reward_exp3m + (reward - mean_reward_exp3m) / (t + 1)
    cumulative_reward_exp3m[t] = mean_reward_exp3m

    t += 1

    if t >= N:
        break

# Implementing Exp3
gamma_exp3 = 0.07

numPlays_exp3 = [1] * N

mean_reward_exp3 = 0
cumulative_reward_exp3 = np.zeros(N)

t = 0


for (choice, reward, est, weights) in exp3m(
    d,
    lambda t, choice: [reward_vector.values[t, i] for i in choice],
    gamma_exp3,
    numPlays_exp3,
):
    mean_reward_exp3 = mean_reward_exp3 + (reward - mean_reward_exp3) / (t + 1)
    cumulative_reward_exp3[t] = mean_reward_exp3

    t += 1

    if t >= N:
        break

# Implementing UCB
ucb_a.run()

# Implementing Eps greedy
eps.run()

# Plotting
plt.figure(figsize=(8.5, 6))
plt.plot(eps.reward, label="$\epsilon-greedy$")
plt.plot(ucb_a.reward, label="UCB")
plt.plot(cumulative_reward_exp3, label="Exp3")
plt.plot(cumulative_reward_exp3m, label="Exp3.M")
plt.plot(cumulative_reward_exp3m_vp, label="Exp3.M-VP")
plt.legend(loc="best")
plt.xlabel("Iterations")
plt.ylabel("Average Reward")
plt.title("Average Rewards")
plt.grid(True)
plt.savefig(plot_dir + "\\compare.png")
_ = plt.show()
