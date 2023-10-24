import numpy as np
from switchingBanditEnv import SwitchingBanditEnv


class EpsilonGreedy:

    def __init__(self, epsilon: float, num_arms: int, possible_rewards):
        self.epsilon = epsilon
        self.num_arms = num_arms
        self.possible_rewards = possible_rewards
        self.counts = [0 for _ in range(self.num_arms)]
        self.values = [0.0 for _ in range(self.num_arms)]

    # Epsilon greedy arm selection
    def choose_arm(self):
        # If prob is not in epsilon, do exploitation of best arm so far
        if np.random.random() > self.epsilon:
            return np.argmax(self.values)
        # If prob falls in epsilon range, do exploration
        else:
            return np.random.randint(low=0, high=self.num_arms)

    # Choose to update chosen arm and reward
    def update(self, chosen_arm, reward):
        # update counts pulled for chosen arm
        self.counts[chosen_arm] = self.counts[chosen_arm] + 1
        n = self.counts[chosen_arm]

        # Update average/mean value/reward for chosen arm
        value = self.values[chosen_arm]
        new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward
        self.values[chosen_arm] = new_value
        return

