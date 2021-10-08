import math
import random

import matplotlib.pyplot as plt
import numpy as np

from exp3M import exp3m
from probability import distr_multiPlays
from utilities import randomInt

# perform the Exp3.M algorithm, which is a mutli-play version of Exp3;
# numActions: number of locations, indexed from 0;
# reward_multiPlays: function accepting as input the multiple locations and producing as output the reward vector for the location set;
# gamma: egalitarianism factor;
# numPlays: number of arms played at each time


# Test Exp3.M using stochastic payoffs for 10 actions with 3 plays at each time.
def simpleTest():
    numActions = 10
    numRounds = 20000
    numPlays_LB = 1
    numPlays_UB = 3
    numPlays_std = 0.9

    numPlays = randomInt(numPlays_LB, numPlays_UB, numPlays_std, numRounds)

    biases = [1.5 / k for k in range(2, 2 + numActions)]
    rewardVector = [
        [1 if random.random() < bias else 0 for bias in biases]
        for _ in range(numRounds)
    ]

    def rewards(t, choice):
        return [rewardVector[t][i] for i in choice]

    # if (numPlays > 1):
    # rewards = lambda t, choice: [rewardVector[t][i] for i in choice]

    # else:
    #     rewards = lambda t, choice: rewardVector[t][choice]

    bestActionSet = sorted(
        range(numActions),
        key=lambda action: sum([rewardVector[t][action] for t in range(numRounds)]),
        reverse=True,
    )[:numPlays_UB]

    gamma = min(
        [
            1,
            math.sqrt(
                numActions
                * numActions
                * numPlays_LB
                * math.log(numActions / numPlays_UB)
                / (((math.e - 2) * numPlays_UB + numPlays_LB) * numPlays_UB * numRounds)
            ),
        ]
    )
    gamma = 0.15

    cumulativeReward = 0
    bestActionCumulativeReward = 0
    weakRegret = 0
    weakRegretVec = []
    linear_regret = []
    regret_Bound_vec = []
    weights_vec = []
    factor = []
    average_reward = []

    t = 0
    for (choice, reward, est, weights) in exp3m(numActions, rewards, gamma, numPlays):
        cumulativeReward += reward
        average_reward.extend([cumulativeReward / (t + 1)])
        bestActionCumulativeReward += sum(
            [rewardVector[t][i] for i in bestActionSet[: numPlays[t]]]
        )
        # bestUpperBoundEstimate = (math.e - 1) * gamma * bestActionCumulativeReward

        weakRegret = bestActionCumulativeReward - cumulativeReward
        averageRegret = weakRegret / (t + 1)
        weights_vec.append(distr_multiPlays(weights, 1))

        # regretBound = (
        #     2
        #     * math.sqrt((1 + (math.e - 2) * numPlays_UB / numPlays_LB))
        #     * math.sqrt(
        #         numPlays_UB * numActions * math.log(numActions / numPlays_UB) * (t + 1)
        #     )
        # )
        regretBound = (
            (1 + (math.e - 2) * numPlays_UB / numPlays_LB)
            * gamma
            * bestActionCumulativeReward
            + (numActions * math.log(numActions / numPlays_UB)) / gamma
        )

        factor.append(weakRegret / regretBound)

        weakRegretVec.append(weakRegret)
        linear_regret.append(t + 1)
        regret_Bound_vec.append(regretBound)

        # print("regret: %d\tmaxRegret: %.2f\taverageRegret: %.2f\tweights: (%s)" % (weakRegret, regretBound,
        # averageRegret, ', '.join(["%.3f" % weight for weight in distr_multiPlays(weights, numPlays[t])])))
        print(
            "regret: %d\tmaxRegret: %.2f\taverRegret: %.2f\tMt: %d\tchoice: (%s)"
            % (
                weakRegret,
                regretBound,
                averageRegret,
                numPlays[t],
                ", ".join(["%.d" % c for c in choice]),
            )
        )

        t += 1

        if t >= numRounds:
            break

    print(cumulativeReward)

    # plotting
    fig1, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(8, 6), dpi=100)
    plt.ylabel("Cumulative (weak) Regret")

    ax1.plot(range(numRounds), weakRegretVec, label="Exp3.M-VP regret", linewidth=3)
    # ax1.plot(range(numRounds), linear_regret, label='linear regret')
    ax1.plot(
        range(numRounds), regret_Bound_vec, label="expected upper bound", linewidth=3
    )
    ax1.legend()
    plt.grid()
    plt.show()

    np_weights_vec = np.array(weights_vec)
    transpose = np_weights_vec.T
    weights_vec = transpose.tolist()

    fig2, ax2 = plt.subplots(nrows=1, ncols=1, figsize=(8, 6), dpi=100)
    for w in weights_vec:
        ax2.plot(range(numRounds), w)

    plt.ylabel("Weight")

    fig2, (ax3, ax4) = plt.subplots(nrows=2, ncols=1, figsize=(8, 6), dpi=100)
    ax3.plot(range(numRounds), factor, label="regret/upper bound", linewidth=3)
    # ax3.set_title("weak regret/upper bound")
    ax3.legend()

    ax4.plot(range(numRounds), average_reward, label="average reward", linewidth=3)
    # ax4.set_title("average reward")
    ax4.legend()


if __name__ == "__main__":
    simpleTest()
