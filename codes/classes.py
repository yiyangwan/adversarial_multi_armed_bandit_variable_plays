import math


class summary:

    choice_p = []
    choice_e = []
    rewardVector_p = []
    rewardVector_e = []
    numPlays = []

    cumulativeReward_p = 0
    cumulativeRewardVec_p = []
    cumulativeReward_e = 0
    cumulativeRewardVec_e = []

    weights_pursuer = []
    weights_evader = []
    dist_evader = []

    bestActionCumulativeReward_p = 0
    bestActionCumulativeReward_e = 0

    weakRegret_p = []
    weakRegret_e = []

    regretBound_p = []
    regretBound_e = []

    bestAction_p = []
    bestAction_e = []


class State:
    def updateState_p(self):
        numActions = len(self.scaledReward)
        numPlays = len(self.choice)
        self.estimatedReward = [0.0] * numActions

        for i in self.choice:
            self.estimatedReward[i] = 1.0 * self.scaledReward[i] / self.probDist[i]

        w_temp_p = self.weights

        for i in range(numActions):
            self.weights[i] *= math.exp(
                numPlays * self.estimatedReward[i] * self.gamma / numActions
            )  # important that we use estimated reward here!

        for s in self.S_0:
            self.weights[s] = w_temp_p[s]

    def updateState_e(self):
        numActions = len(self.scaledReward)
        self.estimatedReward = [0.0] * numActions
        self.estimatedReward[self.choice] = (
            1.0 * self.scaledReward[self.choice] / self.probDist[self.choice]
        )
        self.weights[self.choice] *= math.exp(
            self.estimatedReward[self.choice] * self.gamma / numActions
        )  # important that we use estimated reward here!
