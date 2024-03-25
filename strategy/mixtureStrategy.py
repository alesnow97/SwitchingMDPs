import numpy as np


class MixtureStrategy:

    def __init__(self,
                 policy,
                 num_states,
                 num_actions,
                 num_obs,
                 pomdp,
                 sample_reuse=False,
                 ):

        self.policy = policy
        self.num_states = num_states
        self.num_actions = num_actions
        self.num_obs = num_obs
        self.pomdp = pomdp
        self.sample_reuse = sample_reuse

        self.full_obs_count_vec = np.zeros(shape=(self.num_states, self.num_states,
                                    self.num_actions, self.num_actions,
                                    self.num_obs, self.num_obs))

        self.part_obs_count_vec = np.zeros(shape=(self.num_actions, self.num_actions,
                                    self.num_obs, self.num_obs))


    # the values of samples to discard and samples per estimate refer to the number of couples,
    #  thus the timesteps need to be doubled
    def run(self, num_samples_to_discard, num_samples_per_estimate, num_estimates, initial_state):
        first_state = initial_state

        for estim in range(num_estimates):

            for i in range(num_samples_to_discard+num_samples_per_estimate):

                first_action = self.policy.choose_action()
                first_obs = np.random.multinomial(
                    n=1, pvals=self.pomdp.state_action_observation_matrix[first_state, first_action],
                    size=1)[0].argmax()

                second_state = np.random.multinomial(
                    n=1, pvals=self.pomdp.state_action_transition_matrix[first_state, first_action], size=1)[
                    0].argmax()

                self.policy.update(first_action, first_obs)

                second_action = self.policy.choose_action()
                second_obs = np.random.multinomial(
                    n=1, pvals=self.pomdp.state_action_observation_matrix[
                        second_state, second_action],
                    size=1)[0].argmax()

                next_state = np.random.multinomial(
                    n=1, pvals=self.pomdp.state_action_transition_matrix[
                        second_state, second_action], size=1)[
                    0].argmax()

                if i % 10000 == 0:
                    print(i)

                if i >= num_samples_to_discard:
                    self.full_obs_count_vec[first_state, second_state, first_action, second_action, first_obs, second_obs] += 1
                    self.part_obs_count_vec[first_action, second_action, first_obs, second_obs] += 1

                first_state = next_state


            estimated_trans_matrix = self.estimate_transition_matrix()
            self.policy.update_transition_matrix(
                estimated_transition_matrix=estimated_trans_matrix)

            if self.sample_reuse is False:
                self.full_obs_count_vec = np.zeros(
                    shape=(self.num_states, self.num_states,
                           self.num_actions, self.num_actions,
                           self.num_obs, self.num_obs))

                self.part_obs_count_vec = np.zeros(
                    shape=(self.num_actions, self.num_actions,
                           self.num_obs, self.num_obs))


    def estimate_transition_matrix(self):

        count_vector = self.part_obs_count_vec

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
        probability_matrix_estimation_error = abs(np.sum(distance_matrix))
        print(f"Distance vector is {probability_matrix_estimation_error}")

        print("Real transition matrix is")
        print(self.pomdp.state_action_transition_matrix)

        print("Estimated transition matrix is")
        print(reshaped_transition_matrix)

        return reshaped_transition_matrix