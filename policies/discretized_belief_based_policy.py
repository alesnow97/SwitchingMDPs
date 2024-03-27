from abc import ABC

import numpy as np

import utils
from policies.policy import Policy


class DiscretizedBeliefBasedPolicy(Policy, ABC):

    def __init__(self,
                 num_states,
                 num_actions,
                 num_obs,
                 initial_discretized_belief,
                 initial_discretized_belief_index,
                 discretized_beliefs,
                 estimated_state_action_transition_matrix,
                 belief_action_dist_mapping,
                 state_action_observation_matrix,
                 no_info=False
                 ):
        self.num_states = num_states
        self.num_actions = num_actions
        self.num_obs = num_obs

        self.discretized_beliefs = discretized_beliefs
        self.discretized_belief = initial_discretized_belief
        self.discretized_belief_index = initial_discretized_belief_index

        self.belief_based_policy = belief_action_dist_mapping

        self.state_action_transition_matrix = estimated_state_action_transition_matrix
        self.state_action_observation_matrix = state_action_observation_matrix

        self.no_info = no_info

    def choose_action(self):

        if self.no_info is True:
            chosen_action = np.random.multinomial(
                n=1, pvals=(np.ones(shape=self.num_actions) / self.num_actions),
                size=1)[0].argmax()
        else:
            action_probs = self.belief_based_policy[self.discretized_belief_index, :]
            chosen_action = np.random.multinomial(
                n=1, pvals=action_probs,
                size=1)[0].argmax()

        return chosen_action


    def update(self, action, observation):
        self.discretized_belief_update(
            pulled_arm=action,
            observation_index=observation)

    def discretized_belief_update(self, pulled_arm, observation_index):
        if self.no_info is True:
            return
        observation_distribution = self.state_action_observation_matrix[:,
                                   pulled_arm,
                                   observation_index].reshape(-1)

        scaled_belief = self.discretized_belief * observation_distribution
        current_transition_matrix = self.state_action_transition_matrix[:, pulled_arm, :]
        transitioned_belief = scaled_belief @ current_transition_matrix.T
        belief = transitioned_belief / transitioned_belief.sum()

        self.discretized_belief, self.discretized_belief_index = (
            utils.find_closest_discretized_belief(
                self.discretized_beliefs, belief))

    def continuous_update(self, belief_value, pulled_arm, observation_index):
        current_belief = belief_value
        observation_distribution = self.state_action_observation_matrix[:,
                                   pulled_arm,
                                   observation_index].reshape(-1)

        scaled_belief = current_belief * observation_distribution
        current_transition_matrix = self.state_action_transition_matrix[:, pulled_arm, :]
        transitioned_belief = scaled_belief @ current_transition_matrix.T
        belief = transitioned_belief / transitioned_belief.sum()
        return belief

    def update_policy_infos(self,
                            state_action_transition_matrix,
                            belief_action_dist_mapping):
        self.state_action_transition_matrix = state_action_transition_matrix
        self.belief_based_policy = belief_action_dist_mapping
        self.no_info = False


    def update_belief_from_samples(self, action_obs_samples: list):
        current_belief = np.ones(shape=self.num_states) / self.num_states

        for i in range(len(action_obs_samples)):
            current_ac, current_obs = action_obs_samples[0][0], action_obs_samples[0][1]
            current_belief = self.continuous_update(
                belief_value=current_belief,
                pulled_arm=current_ac,
                observation_index=current_obs
            )

        (self.discretized_belief,
         self.discretized_belief_index) = utils.find_closest_discretized_belief(
            discretized_beliefs=self.discretized_beliefs,
            continuous_belief=current_belief
        )

    def update_transition_matrix(self, transition_matrix):
        return None


