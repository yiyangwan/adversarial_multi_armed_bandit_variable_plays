import math
import random

import numpy as np
import scipy.stats as ss


def getAlpha(temp, w_sorted):
    # getAlpha calculates the alpha value for the sorted weight vector.
    sum_weight = sum(w_sorted)

    for i in range(len(w_sorted)):
        alpha = (temp * sum_weight) / (1 - i * temp)
        curr = w_sorted[i]

        if alpha > curr:
            alpha_exp = alpha
            return alpha_exp

        sum_weight = sum_weight - curr

    raise Exception("alpha not found")


def find_indices(lst, condition):
    # Function that returns the indices satisfying the condition function
    return [i for i, elem in enumerate(lst) if condition(elem)]


def reward(choice_p, choice_e):
    common_choice = list(set(choice_p).intersection([choice_e]))
    reward_e = 0.0
    reward_p = [0.0] * len(choice_p)

    if not common_choice:
        reward_e = 1.0
    else:
        for i in common_choice:
            indx = choice_p.index(i)
            reward_p[indx] = 1.0
    assert sum(reward_p) + reward_e == 1, "Error, should be constant sum game"
    return reward_p, reward_e, common_choice


def DepRound(weights_p, k=1):
    """ [[Algorithms for adversarial bandit problems with multiple plays, by T.Uchiya, A.Nakamura and M.Kudo, 2010](http://hdl.handle.net/2115/47057)] Figure 5 (page 15) is a very clean presentation of the algorithm.

    - Inputs: :math:`k < K` and weights_p :math:`= (p_1, dots, p_K)` such that :math:`sum_{i=1}^{K} p_i = k` (or :math:`= 1`).
    - Output: A subset of :math:`{1,dots,K}` with exactly :math:`k` elements. Each action :math:`i` is selected with probability exactly :math:`p_i`.

    """
    p = np.array(weights_p)
    K = len(p)
    # Checks
    assert k < K, "Error (DepRound): k = {} should be < K = {}.".format(k, K)  # DEBUG

    if not np.isclose(np.sum(p), 1):
        p = p / np.sum(p)
    assert np.all(0 <= p) and np.all(
        p <= 1
    ), "Error: the weights (p_1, ..., p_K) should all be 0 <= p_i <= 1 ...(={})".format(
        p
    )  # DEBUG
    assert np.isclose(
        np.sum(p), 1
    ), "Error: the sum of weights p_1 + ... + p_K should be = 1 (= {}).".format(
        np.sum(p)
    )  # DEBUG

    data = np.array([i for i in range(K)])
    subset = np.random.choice(data, size=k, replace=False, p=p)

    return subset.tolist()


def randomInt(a, b, std, size):
    """ Generate random integers with trancated Gaussian distribution
	- Inputs: lower bound integer "a", upper bound integer "b", standard deviation "std", sample size "size"
	- Output: list of integers "num" selected from interval [a,b] with Gaussian distribution		
	"""
    assert isinstance(a, int) & isinstance(
        b, int
    ), "Error: a and b should be two integers!"
    c = (a + b) * 0.5
    x = np.arange(a, b + 1)
    xU, xL = x + 0.5, x - 0.5
    prob = ss.norm.cdf(xU, loc=c, scale=std) - ss.norm.cdf(xL, loc=c, scale=std)
    prob = prob / prob.sum()  # normalize the probabilities so their sum is 1
    nums = np.random.choice(x, size=size, p=prob)

    return nums.tolist()
