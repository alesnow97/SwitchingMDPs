import numpy as np

import utils
import time

from policies.discretized_belief_based_policy import \
    DiscretizedBeliefBasedPolicy
from strategy import strategy_helper


class OracleStrategy:

    def __init__(self,
                 num_states,
                 num_actions,
                 num_obs,
                 pomdp,
                 ext_v_i_stopping_cond=0.02,
                 epsilon_state=0.2,
                 epsilon_action=0.1,
                 min_action_prob=0.1,
                 ):

        self.num_states = num_states
        self.num_actions = num_actions
        self.num_obs = num_obs
        self.pomdp = pomdp
        self.ext_v_i_stopping_cond = ext_v_i_stopping_cond
        self.epsilon_state = epsilon_state
        self.epsilon_action = epsilon_action
        self.min_action_prob = min_action_prob

        self.discretized_belief_states = utils.discretize_continuous_space(self.num_states, epsilon=epsilon_state)
        self.discretized_action_space = utils.discretize_continuous_space(self.num_actions, epsilon=epsilon_action,
                                                                          min_value=min_action_prob)
        self.len_discretized_beliefs = self.discretized_belief_states.shape[0]
        self.len_discretized_action_space = self.discretized_action_space.shape[0]

        self.real_belief_action_belief = strategy_helper.compute_belief_action_belief_matrix(
            num_actions=self.num_actions,
            num_obs=self.num_obs,
            discretized_belief_states=self.discretized_belief_states,
            len_discretized_beliefs=self.len_discretized_beliefs,
            state_action_transition_matrix=self.pomdp.state_action_transition_matrix,
            state_action_observation_matrix=self.pomdp.state_action_observation_matrix
        )

        self.real_optimal_belief_action_mapping = strategy_helper.compute_optimal_POMDP_policy(
            num_actions=self.num_actions,
            discretized_belief_states=self.discretized_belief_states,
            discretized_action_space=self.discretized_action_space,
            len_discretized_beliefs=self.len_discretized_beliefs,
            len_discretized_action_space=self.len_discretized_action_space,
            ext_v_i_stopping_cond=self.ext_v_i_stopping_cond,
            state_action_reward=self.pomdp.state_action_reward,
            belief_action_belief_matrix=self.real_belief_action_belief
        )

        uniform_initial_belief = np.ones(shape=self.num_states) / self.num_states
        initial_discretized_belief, initial_discretized_belief_index = utils.find_closest_discretized_belief(
            self.discretized_belief_states, uniform_initial_belief
        )

        self.oracle_policy = DiscretizedBeliefBasedPolicy(
            num_states=self.num_states,
            num_actions=self.num_actions,
            num_obs=self.num_obs,
            initial_discretized_belief=initial_discretized_belief,
            initial_discretized_belief_index=initial_discretized_belief_index,
            discretized_beliefs=self.discretized_belief_states,
            estimated_state_action_transition_matrix=self.pomdp.state_action_transition_matrix,
            belief_action_dist_mapping=self.real_optimal_belief_action_mapping,
            state_action_observation_matrix=self.pomdp.state_action_observation_matrix,
            no_info=False
        )


    def run(self, T_0, num_episodes, initial_state):
        current_state = initial_state

        # self.estimated_action_state_dist_per_episode = np.zeros(shape=(
        #     num_episodes, self.num_actions, self.num_actions,
        #     self.num_states, self.num_states))

        self.collected_samples = []

        for i in range(num_episodes):

            last_state = self.collect_samples_in_episode(
                starting_state=current_state,
                T_0=T_0,
                episode_num=i
            )



            current_state = last_state

        return self.collected_samples
        #return self.estimated_action_state_dist, self.estimated_transition_matrix, self.error_frobenious_norm


    def collect_samples_in_episode(self, starting_state, T_0, episode_num):
        # for convenience these numbers are even
        num_samples_to_discard = int(np.log(T_0 * 2**episode_num))
        num_samples_in_episode = int(T_0 * 2**episode_num)

        self.full_obs_count_vec = np.zeros(
            shape=(self.num_states, self.num_states,
                   self.num_actions, self.num_actions,
                   self.num_obs, self.num_obs))

        self.part_obs_count_vec = np.zeros(
            shape=(self.num_actions, self.num_actions,
                   self.num_obs, self.num_obs))

        num_total_samples = num_samples_to_discard + num_samples_in_episode
        first_state = starting_state

        for sample_num in range(num_total_samples):

            first_action = self.oracle_policy.choose_action()
            first_obs = np.random.multinomial(
                n=1, pvals=self.pomdp.state_action_observation_matrix[
                    first_state, first_action],
                size=1)[0].argmax()

            second_state = np.random.multinomial(
                n=1, pvals=self.pomdp.state_action_transition_matrix[
                    first_state, first_action], size=1)[
                0].argmax()

            self.oracle_policy.update(first_action, first_obs)
            self.collected_samples.append((first_action, first_obs))

            second_action = self.oracle_policy.choose_action()
            second_obs = np.random.multinomial(
                n=1, pvals=self.pomdp.state_action_observation_matrix[
                    second_state, second_action],
                size=1)[0].argmax()

            next_state = np.random.multinomial(
                n=1, pvals=self.pomdp.state_action_transition_matrix[
                    second_state, second_action], size=1)[
                0].argmax()

            self.oracle_policy.update(second_action, second_obs)
            self.collected_samples.append((second_action, second_obs))

            if sample_num % 1000 == 0:
                print(sample_num)

            first_state = next_state

        return first_state

