import numpy as np

import utils

from policies.discretized_belief_based_policy import \
    DiscretizedBeliefBasedPolicy
from strategy import strategy_helper


class OptimisticAlgorithmStrategy:

    def __init__(self,
                 num_states,
                 num_actions,
                 num_obs,
                 pomdp,
                 ext_v_i_stopping_cond=0.02,
                 epsilon_state=0.2,
                 epsilon_action=0.1,
                 min_action_prob=0.1,
                 delta=0.1
                 ):

        self.num_states = num_states
        self.num_actions = num_actions
        self.num_obs = num_obs
        self.pomdp = pomdp
        self.ext_v_i_stopping_cond = ext_v_i_stopping_cond
        self.epsilon_state = epsilon_state
        self.epsilon_action = epsilon_action
        self.min_action_prob = min_action_prob
        self.delta = delta

        self.discretized_belief_states = utils.discretize_continuous_space(self.num_states, epsilon=epsilon_state)
        self.discretized_action_space = utils.discretize_continuous_space(self.num_actions, epsilon=epsilon_action,
                                                                          min_value=min_action_prob)
        self.len_discretized_beliefs = self.discretized_belief_states.shape[0]
        self.len_discretized_action_space = self.discretized_action_space.shape[0]

    # the values of samples to discard and samples per estimate refer to the number of couples,
    #  thus the timesteps need to be doubled
    def run(self, T_0, num_episodes, initial_state):
        current_state = initial_state

        # self.estimated_action_state_dist_per_episode = np.zeros(shape=(
        #     num_episodes, self.num_actions, self.num_actions,
        #     self.num_states, self.num_states))

        self.init_policy()
        self.collected_samples = []

        self.estimated_transition_matrix_per_episode = np.zeros(shape=(
            num_episodes, self.num_states, self.num_actions,
            self.num_states))

        self.error_frobenious_norm_per_episode = np.empty(shape=num_episodes)

        for i in range(num_episodes):

            estimated_transition_matrix, frobenious_norm = (
                self.collect_samples_in_episode(
                starting_state=current_state,
                T_0=T_0,
                episode_num=i
            ))

            self.estimated_transition_matrix_per_episode[i] = estimated_transition_matrix
            self.error_frobenious_norm_per_episode[i] = frobenious_norm

            current_confidence_bound = self.compute_confidence_bound(T_0=T_0, episode_num=i)

            # compute optimistic mdp
            optimistic_transition_matrix_mdp, optimistic_policy_mdp = (
                strategy_helper.compute_optimistic_MDP(
                    num_states=self.num_states,
                    num_actions=self.num_actions,
                    discretized_action_space=self.discretized_action_space,
                    state_action_transition_matrix=estimated_transition_matrix,
                    ext_v_i_stopping_cond=self.ext_v_i_stopping_cond,
                    state_action_reward=self.pomdp.state_action_reward,
                    confidence_bound=current_confidence_bound,
                    min_transition_value=self.pomdp.min_transition_value
                ))

            # compute belief action belief matrix from the optimistic mdp
            optimistic_belief_action_belief_matrix = (
                strategy_helper.compute_belief_action_belief_matrix(
                    num_actions=self.num_actions,
                    num_obs=self.num_obs,
                    discretized_belief_states=self.discretized_belief_states,
                    len_discretized_beliefs=self.len_discretized_beliefs,
                    state_action_transition_matrix=optimistic_transition_matrix_mdp,
                    state_action_observation_matrix=self.pomdp.state_action_observation_matrix
                ))

            optimistic_belief_action_mapping = strategy_helper.compute_optimal_POMDP_policy(
                num_actions=self.num_actions,
                discretized_belief_states=self.discretized_belief_states,
                discretized_action_space=self.discretized_action_space,
                len_discretized_beliefs=self.len_discretized_beliefs,
                len_discretized_action_space=self.len_discretized_action_space,
                ext_v_i_stopping_cond=self.ext_v_i_stopping_cond,
                state_action_reward=self.pomdp.state_action_reward,
                belief_action_belief_matrix=optimistic_belief_action_belief_matrix
            )

            self.policy.update_policy_infos(
                state_action_transition_matrix=optimistic_transition_matrix_mdp,
                belief_action_dist_mapping=optimistic_belief_action_mapping
            )

            num_last_samples_for_belief_update = 40 + int(np.log(T_0 * 2**i))
            self.policy.update_belief_from_samples(
                action_obs_samples=self.collected_samples[
                                   -num_last_samples_for_belief_update:]
            )

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

            first_action = self.policy.choose_action()
            first_obs = np.random.multinomial(
                n=1, pvals=self.pomdp.state_action_observation_matrix[
                    first_state, first_action],
                size=1)[0].argmax()

            second_state = np.random.multinomial(
                n=1, pvals=self.pomdp.state_action_transition_matrix[
                    first_state, first_action], size=1)[
                0].argmax()

            self.policy.update(first_action, first_obs)
            self.collected_samples.append((first_action, first_obs))

            second_action = self.policy.choose_action()
            second_obs = np.random.multinomial(
                n=1, pvals=self.pomdp.state_action_observation_matrix[
                    second_state, second_action],
                size=1)[0].argmax()

            next_state = np.random.multinomial(
                n=1, pvals=self.pomdp.state_action_transition_matrix[
                    second_state, second_action], size=1)[
                0].argmax()

            self.policy.update(second_action, second_obs)
            self.collected_samples.append((second_action, second_obs))

            if sample_num % 10000 == 0:
                print(sample_num)

            if sample_num >= num_samples_to_discard:
                self.full_obs_count_vec[
                    first_state, second_state, first_action, second_action, first_obs, second_obs] += 1
                self.part_obs_count_vec[
                    first_action, second_action, first_obs, second_obs] += 1

            first_state = next_state

        return self.estimate_transition_matrix()
        # self.policy.update_transition_matrix(
        #     estimated_transition_matrix=estimated_trans_matrix)


    def compute_confidence_bound(self, T_0, episode_num):
        min_transition_value = self.pomdp.state_action_transition_matrix.min()
        min_action_prob = self.min_action_prob
        # lambda_max = 1 - self.num_states*self.num_actions*min_transition_value*min_action_prob
        # print(lambda_max)

        alpha = self.pomdp.min_svd_reference_matrix

        eps_iota_sq = np.power(min_transition_value * min_action_prob, 3/2)

        # first_term = 4 / (alpha**2 * eps_iota_sq)
        first_term = 4 / np.sqrt(eps_iota_sq)
        second_term = np.sqrt(1 + np.log(np.max((episode_num, 1))**3/self.delta)) / np.sqrt(T_0 * 2**episode_num)

        # this is the bound in frobenious norm
        confidence_bound = first_term * second_term
        # confidence_bound = second_term
        print(f"Confidence bound is {confidence_bound}")

        # TODO sistemare il fatto che usiamo
        #  la norma frobenious rispetto alla norma 1

        return confidence_bound

    def estimate_transition_matrix(self):

        count_vector = self.part_obs_count_vec.copy()

        # estimate d(a,a',o,o') from count vector
        count_vector = count_vector.reshape(-1)
        act_obs_pair_dist = count_vector / count_vector.sum()

        # compute d(a,a',s,s')
        act_stat_pair_dist = np.linalg.lstsq(self.pomdp.reference_matrix, act_obs_pair_dist, rcond=None)[0]

        act_stat_pair_dist = act_stat_pair_dist.reshape(
            (self.num_actions, self.num_actions, self.num_states, self.num_states))

        reduced_act_stat_pair_dist = act_stat_pair_dist.sum(axis=1)
        sum_over_last_state = reduced_act_stat_pair_dist.sum(axis=2)

        # this is expressed in the form action, state, state
        estimated_transition_matrix = reduced_act_stat_pair_dist / sum_over_last_state[:, :, None]
        reshaped_transition_matrix = np.empty(shape=(self.num_states, self.num_actions, self.num_states))
        for action in range(self.num_actions):
            reshaped_transition_matrix[:, action, :] = estimated_transition_matrix[action, :, :]

        distance_matrix = np.absolute(self.pomdp.state_action_transition_matrix.reshape(-1) -
            reshaped_transition_matrix.reshape(-1))

        frobenious_norm = np.sqrt(np.sum(distance_matrix**2))

        probability_matrix_estimation_error = abs(np.sum(distance_matrix))
        print(f"Distance vector norm-1 is {probability_matrix_estimation_error}")
        print(
            f"Distance vector, frobenious norm, is {frobenious_norm}")

        print("Real transition matrix is")
        print(self.pomdp.state_action_transition_matrix)

        print("Estimated transition matrix is")
        print(reshaped_transition_matrix)

        # fix the transition matrix if some negative numbers are present
        if np.any(reshaped_transition_matrix <= 0):
            modified_transition_matrix = reshaped_transition_matrix.copy()
            counter = 1
            while np.any(modified_transition_matrix < self.pomdp.min_transition_value - 0.05):
                modified_transition_matrix[
                    modified_transition_matrix <= self.pomdp.min_transition_value] = self.pomdp.min_transition_value + 0.02 * counter
                modified_transition_matrix[
                    modified_transition_matrix >= (
                                1 - self.pomdp.non_normalized_min_transition_value)] = 1 - self.pomdp.non_normalized_min_transition_value
                sum_over_last_state = modified_transition_matrix.sum(axis=2)
                modified_transition_matrix = (modified_transition_matrix /
                                              sum_over_last_state[:, :, None])
                counter += 1
            reshaped_transition_matrix = modified_transition_matrix

        return reshaped_transition_matrix, frobenious_norm


    def init_policy(self):

        # used policy
        self.policy = DiscretizedBeliefBasedPolicy(
            num_states=self.num_states,
            num_actions=self.num_actions,
            num_obs=self.num_obs,
            initial_discretized_belief=None,
            initial_discretized_belief_index=None,
            discretized_beliefs=self.discretized_belief_states,
            estimated_state_action_transition_matrix=None,
            belief_action_dist_mapping=None,
            state_action_observation_matrix=self.pomdp.state_action_observation_matrix,
            no_info=True
        )