import math
import os
import random
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import yaml

from classes import State, summary
from probability import distr, distr_multiPlays, draw
from utilities import DepRound, find_indices, getAlpha, randomInt, reward
from utility_plot import regret_plot

sys.path.append(".")


def advGame(config, numPlays):
    numActions = config["numActions"]
    numRounds = config["numRounds"]

    rewardMax = config["rewardMax"]
    rewardMin = config["rewardMin"]

    State_p = State()
    State_e = State()

    State_p.weights = [1.0] * numActions  # initialize weight vector for the pursuer
    State_e.weights = [1.0] * numActions  # initialize weight vector for the evader

    t = 0

    if config["custom_gamma_pursuer"]:
        State_p.gamma = config["gamma_p"]

    else:
        State_p.gamma = min(
            [
                1,
                math.sqrt(
                    numActions
                    * numActions
                    * numPlays_LB
                    * math.log(numActions / numPlays_UB)
                    / (
                        ((math.e - 2) * numPlays_UB + numPlays_LB)
                        * numPlays_UB
                        * numRounds
                    )
                ),
            ]
        )

    if config["custom_gamma_evader"]:
        State_e.gamma = min(
            [
                1,
                math.sqrt(
                    numActions * math.log(numActions) / ((math.e - 1) * numRounds)
                ),
            ]
        )

    else:
        State_e.gamma = config["gamma_e"]

    # entering the main loop
    while True:
        theSum_pursuer = sum(State_p.weights)
        theSum_evader = sum(State_e.weights)
        State_p.weights = [
            w / theSum_pursuer for w in State_p.weights
        ]  # normalize the weight vector of the pursuer
        State_e.weights = [
            w / theSum_evader for w in State_e.weights
        ]  # normalize the weight vector of the pursuer

        temp = (1.0 / numPlays - State_p.gamma / numActions) * float(
            1.0 / (1.0 - State_p.gamma)
        )
        w_temp_p = State_p.weights
        w_sorted_p = sorted(State_p.weights, reverse=True)

        if w_sorted_p[0] >= temp * theSum_pursuer:
            State_p.alpha_t = getAlpha(temp, w_sorted_p)

            State_p.S_0 = find_indices(w_temp_p, lambda e: e >= State_p.alpha_t)

            for s in State_p.S_0:
                w_temp_p[s] = State_p.alpha_t

        else:
            State_p.S_0 = []

        State_p.probDist = distr_multiPlays(w_temp_p, numPlays, gamma=State_p.gamma)
        State_e.probDist = distr(State_e.weights, gamma=State_e.gamma)

        assert False in np.isnan(
            np.array(State_p.probDist)
        ), "Error, probability of pursuer must be a real number"
        assert False in np.isnan(
            np.array(State_e.probDist)
        ), "Error, probability of evader must be a real number"

        State_p.choice = sorted(DepRound(State_p.probDist, k=numPlays))
        State_e.choice = draw(State_e.probDist)

        State_p.reward, State_e.reward, _ = reward(State_p.choice, State_e.choice)

        State_p.rewardFull = [
            0.0
        ] * numActions  # initialize reward vector for the pursuer
        State_e.rewardFull = [
            0.0
        ] * numActions  # initialize reward vector for the evader

        for i in State_p.choice:
            for r in State_p.reward:
                State_p.rewardFull[i] = r

        State_e.rewardFull[State_e.choice] = State_e.reward

        State_p.scaledReward = [
            (r - rewardMin) / (rewardMax - rewardMin) for r in State_p.rewardFull
        ]  # reward vector scaled to 0,1, optional
        State_e.scaledReward = [
            (r - rewardMin) / (rewardMax - rewardMin) for r in State_e.rewardFull
        ]  # reward vector scaled to 0,1, optional

        State_p.updateState_p()
        State_e.updateState_e()

        State_p.cumulativeReward = sum(State_p.reward)
        State_e.cumulativeReward = State_e.reward
        assert (
            State_p.cumulativeReward + State_e.cumulativeReward == 1
        ), "Error, should be constant sum game!"

        yield State_p, State_e
        t = t + 1


if __name__ == "__main__":
    with open("config_search.yaml", "r+") as f_search:
        config_search = yaml.load(f_search)

    for nA_idx, nA_val in enumerate(config_search["numActionsVec"]):
        config_search["numActions"] = nA_val

        # iterates through all combinations

        for nR_idx, nR_val in enumerate(config_search["numRoundsVec"]):
            for nPL_idx, nPL_val in enumerate(config_search["numPlays_LBVec"]):
                for nPU_idx, nPU_val in enumerate(config_search["numPlays_UBVec"]):

                    config_search["numRounds"] = nR_val
                    config_search["numPlays_LB"] = nPL_val
                    config_search["numPlays_UB"] = nPU_val

                    # create the directories for recording results and summaries
                    subscript = str(nA_idx) + str(nR_idx) + str(nPL_idx) + str(nPU_idx)
                    config_search["summary_dir"] = (
                        config_search["summaries_dir"] + "summary_" + subscript + "/"
                    )
                    config_search["plot_dir"] = (
                        config_search["plots_dir"] + "plot_" + subscript + "/"
                    )

                    if not os.path.exists(config_search["summary_dir"]):
                        os.makedirs(config_search["summary_dir"])

                    if not os.path.exists(config_search["plot_dir"]):
                        os.makedirs(config_search["plot_dir"])

                    with open(config_search["plot_dir"] + "config.yaml", "w") as f:
                        yaml.dump(config_search, f)

                    t = 0

                    numPlays_LB = nPL_val
                    numPlays_UB = nPU_val
                    numPlays_std = config_search["numPlays_std"]
                    # summary.numPlays = [random.randint(numPlays_LB, numPlays_UB) for _ in range(nR_val)] 		# list of number of plays that is uniformly distributed for the pursuer
                    summary.numPlays = randomInt(
                        numPlays_LB, numPlays_UB, numPlays_std, nR_val
                    )

                    try:
                        for (State_p, State_e) in advGame(
                            config_search, summary.numPlays[t]
                        ):
                            summary.cumulativeReward_p += State_p.cumulativeReward
                            summary.cumulativeRewardVec_p.extend(
                                [summary.cumulativeReward_p]
                            )
                            summary.cumulativeReward_e += State_e.cumulativeReward
                            summary.cumulativeRewardVec_e.extend(
                                [summary.cumulativeReward_e]
                            )

                            summary.weights_pursuer.append(State_p.weights)
                            summary.weights_evader.append(State_e.weights)
                            summary.dist_evader.append(distr(State_e.weights))

                            summary.choice_p.append(State_p.choice)
                            summary.choice_e.append([State_e.choice])
                            summary.rewardVector_p.append(State_p.rewardFull)
                            summary.rewardVector_e.append(State_e.rewardFull)

                            print(
                                "pursuer weights:(%s)\tevader weights(%s)\t"
                                % (
                                    ", ".join(
                                        [
                                            "%.3f" % r
                                            for r in distr_multiPlays(
                                                State_p.weights, summary.numPlays[t]
                                            )
                                        ]
                                    ),
                                    ",".join(
                                        ["%.3f" % r for r in distr(State_e.weights)]
                                    ),
                                )
                            )

                            t += 1
                            if t >= nR_val:
                                break

                        bestActionSet_p = sorted(
                            range(config_search["numActions"]),
                            key=lambda action: sum(
                                [
                                    summary.rewardVector_p[t][action]
                                    for t in range(nR_val)
                                ]
                            ),
                            reverse=True,
                        )[: config_search["numPlays_UB"]]
                        bestActionSet_e = max(
                            range(config_search["numActions"]),
                            key=lambda action: sum(
                                [
                                    summary.rewardVector_e[t][action]
                                    for t in range(nR_val)
                                ]
                            ),
                        )

                        log_p = open(
                            config_search["summary_dir"] + "log_pursuer.txt", "w+"
                        )
                        log_e = open(
                            config_search["summary_dir"] + "log_evader.txt", "w+"
                        )

                        for s in range(nR_val):
                            summary.bestAction_p.append(
                                bestActionSet_p[: summary.numPlays[s]]
                            )
                            summary.bestActionCumulativeReward_p += sum(
                                [
                                    summary.rewardVector_p[s][i]
                                    for i in bestActionSet_p[: summary.numPlays[s]]
                                ]
                            )
                            summary.bestActionCumulativeReward_e += summary.rewardVector_e[
                                s
                            ][
                                bestActionSet_e
                            ]

                            summary.regretBound_p.append(
                                (
                                    1
                                    + (math.e - 2)
                                    * config_search["numPlays_UB"]
                                    / config_search["numPlays_LB"]
                                )
                                * State_p.gamma
                                * summary.bestActionCumulativeReward_p
                                + (
                                    config_search["numActions"]
                                    * math.log(
                                        config_search["numActions"]
                                        / config_search["numPlays_UB"]
                                    )
                                )
                                / State_p.gamma
                            )

                            summary.regretBound_e.append(
                                (math.e - 1)
                                * State_e.gamma
                                * summary.bestActionCumulativeReward_e
                                + (
                                    config_search["numActions"]
                                    * math.log(config_search["numActions"])
                                )
                                / State_e.gamma
                            )

                            summary.weakRegret_p.extend(
                                [
                                    summary.bestActionCumulativeReward_p
                                    - summary.cumulativeRewardVec_p[s]
                                ]
                            )
                            summary.weakRegret_e.extend(
                                [
                                    summary.bestActionCumulativeReward_e
                                    - summary.cumulativeRewardVec_e[s]
                                ]
                            )

                            log_p.write(
                                "t:%d\tnumPlays:%d\tregret:%d\treward:%d\tweights:(%s)\r\n"
                                % (
                                    s + 1,
                                    summary.numPlays[s],
                                    summary.weakRegret_p[s],
                                    summary.cumulativeRewardVec_p[s],
                                    ", ".join(
                                        [
                                            "%.3f" % weight
                                            for weight in distr_multiPlays(
                                                summary.weights_pursuer[s],
                                                summary.numPlays[s],
                                            )
                                        ]
                                    ),
                                )
                            )

                            log_e.write(
                                "t:%d\tregret:%d\treward:%d\tweights:(%s)\r\n"
                                % (
                                    s + 1,
                                    summary.weakRegret_e[s],
                                    summary.cumulativeRewardVec_e[s],
                                    ", ".join(
                                        [
                                            "%.3f" % weight
                                            for weight in distr(
                                                summary.weights_evader[s]
                                            )
                                        ]
                                    ),
                                )
                            )

                        regret_plot(summary, config_search)  # ploting

                    except KeyboardInterrupt:
                        pass
