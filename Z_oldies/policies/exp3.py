import numpy as np


class Exp3:

    def __init__(self, num_arms, possible_rewards, gamma: float, alpha: float):
        self.num_arms = num_arms
        self.possible_rewards = possible_rewards
        self.reward_min = possible_rewards.min()
        self.reward_max = possible_rewards.max()
        self.gamma = gamma
        self.weights = np.ones(shape=num_arms)
        self.probabilities = None

    # Epsilon greedy arm selection
    def choose_arm(self):
        self.probabilities = self.compute_probability_distributions()
        chosen_action = np.random.multinomial(1, self.probabilities, 1)[0].argmax()
        return chosen_action

    # Choose to update chosen arm and reward
    def update(self, chosen_arm, reward):
        estimated_reward = 1.0 * reward / self.probabilities[chosen_arm]
        self.weights[chosen_arm] *= np.exp(estimated_reward * self.gamma / self.num_arms)

    def compute_probability_distributions(self):
        probabilities = (1.0 - self.gamma) * self.weights / self.weights.sum()
        additive_term = np.array([self.gamma / self.num_arms] * self.num_arms)
        probabilities += additive_term
        print(probabilities.sum())
        print("Sum of probabilities is the obe defined before")
        return probabilities

