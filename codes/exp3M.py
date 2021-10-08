import math
import random

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fsolve

from probability import distr_multiPlays
from utilities import DepRound, find_indices

# perform the Exp3.M algorithm, which is a mutli-play version of Exp3;
# numActions: number of locations, indexed from 0;
# reward_multiPlays: function accepting as input the multiple locations and producing as output the reward vector for the location set;
# gamma: egalitarianism factor;
# numPlays: number of arms played at each time, must be a vector!!!


def getAlpha(temp, w_sorted):
    # getAlpha calculates the alpha value for the sorted weight.
    sum_weight = sum(w_sorted)

    for i in range(len(w_sorted)):
        alpha = (temp * sum_weight) / (1 - i * temp)
        curr = w_sorted[i]

        if alpha > curr:
            alpha_exp = alpha
            return alpha_exp

        sum_weight = sum_weight - curr
    raise Exception("alpha not found")


def exp3m(numActions, reward_multiPlays, gamma, numPlays, rewardMin=0, rewardMax=1):
    weights = [1.0] * numActions  # initialize weight vector

    t = 0
    while True:
        theSum = sum(weights)
        weights = [w / theSum for w in weights]  # normalize the weight vector
        temp = (1.0 / numPlays[t] - gamma / numActions) * float(1.0 / (1.0 - gamma))
        w_temp1 = weights

        if max(weights) >= temp * theSum:
            # alpha = Symbol('alpha')
            # w_temp = weights
            # w_temp.sort() # sort weights vector in asceding order, optional

            # fun_1 = lambda alpha: alpha / (sum( [alpha for w in weights if w >= alpha] ) + sum( [w for w in weights if w < alpha] ) ) - (1.0 / numPlays - gamma / numActions) / (1.0 - gamma)

            # x_initial = 0                                                 # set initial search point for fsolve solver to find the value alpha_t
            # alpha_t = fsolve(fun_1, x_initial)
            w_sorted = sorted(weights, reverse=True)
            alpha_t = getAlpha(temp, w_sorted)
            # alpha_t = alpha_t.tolist() # convert numpy output by fsolve solver to list

            idx_temp = 0
            S_null = []

            S_null = find_indices(w_temp1, lambda e: e >= alpha_t)

            for s in S_null:
                w_temp1[s] = alpha_t

        else:
            S_null = []

        probabilityDistribution = distr_multiPlays(w_temp1, numPlays[t], gamma=gamma)
        # if True in np.isnan(np.array(probabilityDistribution)):
        #   pass

        assert False in np.isnan(
            np.array(probabilityDistribution)
        ), "Error, probability must be a real number"

        choice = sorted(
            DepRound(probabilityDistribution, k=numPlays[t])
        )  # list of choice
        # choice = DepRound1(probabilityDistribution,k = numPlays[t])                  # list of choice
        theReward = reward_multiPlays(
            t, choice
        )  # input: choice list and time; output: reward list

        theReward_full = [0.0] * numActions  # initialize reward vector

        for i in choice:
            for r in theReward:
                theReward_full[i] = r

        scaledReward = [
            (r - rewardMin) / (rewardMax - rewardMin) for r in theReward_full
        ]  # reward vector scaled to 0,1, optional
        # probabilityChoice = [probabilityDistribution[i] for i in choice]          # probability vector of selected locations

        estimateReward = [0.0] * numActions

        for i in choice:
            estimateReward[i] = 1.0 * scaledReward[i] / probabilityDistribution[i]

        w_temp = weights

        for i in range(numActions):
            weights[i] *= math.exp(
                numPlays[t] * estimateReward[i] * gamma / numActions
            )  # important that we use estimated reward here!

        for s in S_null:
            weights[s] = w_temp[s]

        cumulativeReward = sum(theReward)

        if sum(weights) == 0:
            pass  # Debugging

        yield choice, cumulativeReward, estimateReward, weights
        t = t + 1


# Test Exp3.M using stochastic payoffs for 10 actions with 3 plays at each time.
def simpleTest():
    numActions = 10
    numRounds = 200000
    numPlays = [3] * numRounds

    biases = [1.5 / k for k in range(2, 2 + numActions)]
    rewardVector = [
        [1 if random.random() < bias else 0 for bias in biases]
        for _ in range(numRounds)
    ]

    rewards = lambda t, choice: [rewardVector[t][i] for i in choice]
    # if (numPlays[0] > 1):
    # rewards = lambda t, choice: [rewardVector[t][i] for i in choice]

    # else:
    #        rewards = lambda t, choice: rewardVector[t][choice]

    bestAction = sorted(
        range(numActions),
        key=lambda action: sum([rewardVector[t][action] for t in range(numRounds)]),
        reverse=True,
    )[: numPlays[0]]

    gamma = min(
        [
            1,
            math.sqrt(
                numActions
                * math.log(numActions / numPlays[0])
                / ((math.e - 1) * numPlays[0] * numRounds)
            ),
        ]
    )
    gamma = 0.1

    cumulativeReward = 0
    bestActionCumulativeReward = 0
    weakRegret = 0
    weakRegretVec = []
    linear_regret = []
    regret_Bound_vec = []
    weights_vec = []
    factor = []

    t = 0
    for (choice, reward, est, weights) in exp3m(numActions, rewards, gamma, numPlays):
        cumulativeReward += reward
        bestActionCumulativeReward += sum([rewardVector[t][i] for i in bestAction])
        bestUpperBoundEstimate = (math.e - 1) * gamma * bestActionCumulativeReward

        weakRegret = bestActionCumulativeReward - cumulativeReward
        averageRegret = weakRegret / (t + 1)
        weights_vec.append(distr_multiPlays(weights, numPlays[0]))

        # regretBound = (math.e - 1) * gamma * bestActionCumulativeReward + (numActions * math.log(numActions)) / gamma
        # regretBound = 2.63 * math.sqrt(numActions * t * numPlays[0] * math.log(numActions/numPlays[0]) )

        regretBound = (
            bestUpperBoundEstimate
            + (numActions * math.log(numActions / numPlays[0])) / gamma
        )
        factor.append(weakRegret / regretBound)

        weakRegretVec.append(weakRegret)
        linear_regret.append(t + 1)
        regret_Bound_vec.append(regretBound)

        print(
            "regret: %d\tmaxRegret: %.2f\taverageRegret: %.2f\tweights: (%s)"
            % (
                weakRegret,
                regretBound,
                averageRegret,
                ", ".join(
                    [
                        "%.3f" % weight
                        for weight in distr_multiPlays(weights, numPlays[0])
                    ]
                ),
            )
        )

        t += 1

        if t >= numRounds:
            break

    print(cumulativeReward)

    # plotting
    fig1, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
    plt.ylabel("Cumulative (weak) Regret")
    ax1.plot(range(numRounds), weakRegretVec, label="weak regret")
    ax1.plot(range(numRounds), linear_regret, label="linear regret")
    ax1.plot(range(numRounds), regret_Bound_vec, label="expected upper bound")
    ax1.legend()

    np_weights_vec = np.array(weights_vec)
    transpose = np_weights_vec.T
    weights_vec = transpose.tolist()

    for w in weights_vec:
        ax2.plot(range(numRounds), w)

    plt.ylabel("Weight")

    fig2, ax3 = plt.subplots()
    ax3.plot(range(numRounds), factor, label="weak regret/upper bound")
    ax3.set_title("weak regret/upper bound")


if __name__ == "__main__":
    simpleTest()
