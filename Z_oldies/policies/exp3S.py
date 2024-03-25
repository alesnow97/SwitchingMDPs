import numpy as np


class Exp3S:

    def __init__(self, num_arms, possible_rewards, gamma: float, alpha: float, limit:int, normalization_factor:int):
        self.num_arms = num_arms
        self.possible_rewards = possible_rewards
        self.reward_min = possible_rewards.min()
        self.reward_max = possible_rewards.max()
        self.limit = limit
        self.normalization_factor = normalization_factor
        self.gamma = gamma
        self.alpha = alpha
        self.weights = np.ones(shape=num_arms)
        self.probabilities = None

    # Epsilon greedy arm selection
    def choose_arm(self):
        self.probabilities = self.compute_probability_distributions()
        chosen_action = np.random.multinomial(1, self.probabilities, 1)[0].argmax()
        return chosen_action

    # Choose to update chosen arm and reward
    def update(self, chosen_arm, reward):
        estimated_rewards = np.zeros(shape=self.num_arms)
        estimated_rewards[chosen_arm] = 1.0 * reward / self.probabilities[chosen_arm]

        modified_estimated_rewards = np.exp(estimated_rewards * self.gamma / self.num_arms)
        addendum1 = self.weights * modified_estimated_rewards
        addendum2 = np.array([self.weights.sum() * np.e * self.alpha / self.num_arms] * self.num_arms)
        self.weights = addendum1 + addendum2

        if self.weights.sum() > np.e ** self.limit:
            print("Weights normalized")
            self.weights = self.weights / self.weights.sum() * self.normalization_factor

    def compute_probability_distributions(self):
        probabilities = (1.0 - self.gamma) * self.weights / self.weights.sum()
        additive_term = np.array([self.gamma / self.num_arms] * self.num_arms)
        probabilities += additive_term
        return probabilities

