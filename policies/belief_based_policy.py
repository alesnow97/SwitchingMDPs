from abc import ABC

import numpy as np

from policies.policy import Policy


class BeliefBasedPolicy(Policy, ABC):

    def __init__(self,
                 num_states,
                 num_actions,
                 num_obs,
                 state_action_transition_matrix,
                 state_action_observation_matrix,
                 possible_rewards
                 ):
        self.num_states = num_states
        self.num_actions = num_actions
        self.num_obs = num_obs
        self.state_action_transition_matrix = np.ones(shape=(self.num_states, self.num_actions, self.num_states)) * 1/self.num_states
        self.state_action_observation_matrix = state_action_observation_matrix
        self.possible_rewards = possible_rewards

        self.belief = np.ones(shape=(num_states)) * 1 / num_states


    def choose_action(self):
        current_state_action_observation = self.state_action_observation_matrix

        current_state_action_reward = current_state_action_observation * self.possible_rewards[None, None,
                                        :]

        scaled_matrix = self.belief[:, None,
                        None] * current_state_action_reward

        reduced_scaled_matrix = scaled_matrix.sum(axis=2)
        final_vector = reduced_scaled_matrix.sum(axis=0)
        best_action = np.argmax(final_vector)

        alpha = 0.5
        eps_prob = alpha / self.num_actions

        action_probs = np.ones(shape=(self.num_actions)) * eps_prob
        action_probs[best_action] += 1 - alpha

        chosen_action = np.random.multinomial(
            n=1, pvals=action_probs,
            size=1)[0].argmax()

        return chosen_action

    def update(self, pulled_arm, observation_index):
        observation_distribution = self.state_action_observation_matrix[:,
                                   pulled_arm,
                                   observation_index].reshape(-1)

        scaled_belief = self.belief * observation_distribution
        current_transition_matrix = self.state_action_transition_matrix[:, pulled_arm, :]
        transitioned_belief = scaled_belief @ current_transition_matrix.T

        self.belief = transitioned_belief / transitioned_belief.sum()


    def update_transition_matrix(self, estimated_transition_matrix):
        self.state_action_transition_matrix = estimated_transition_matrix


