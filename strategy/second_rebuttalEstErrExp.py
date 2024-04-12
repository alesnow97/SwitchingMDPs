import numpy as np


class SecondEstimationErrorStrategy:

    def __init__(self,
                 policy,
                 num_states,
                 num_actions,
                 num_obs,
                 pomdp
                 ):

        self.policy = policy
        self.num_states = num_states
        self.num_actions = num_actions
        self.num_obs = num_obs
        self.pomdp = pomdp


    # the values of samples to discard and samples per estimate refer to the number of couples,
    #  thus the timesteps need to be doubled
    def run(self, num_samples_to_discard, num_samples_checkpoint, num_checkpoints, initial_state):
        first_state = initial_state

        self.full_obs_count_vec = np.zeros(shape=(self.num_states, self.num_states,
                                    self.num_actions, self.num_actions,
                                    self.num_obs, self.num_obs))

        self.part_obs_count_vec = np.zeros(shape=(self.num_actions, self.num_actions,
                                    self.num_obs, self.num_obs))

        self.action_state_dist = np.zeros(shape=(self.num_actions, self.num_actions,
                                    self.num_states, self.num_states))

        self.estimated_action_state_dist = np.zeros(shape=(
            num_checkpoints, self.num_actions, self.num_actions,
            self.num_states, self.num_states))

        self.estimated_transition_matrix = np.zeros(shape=(
            num_checkpoints, self.num_states, self.num_actions, self.num_states))

        self.error_frobenious_norm = np.empty(shape=num_checkpoints)

        # self.num_samples_per_estimate = np.zeros(shape=(num_estimates))
        num_total_samples = num_samples_to_discard + num_samples_checkpoint * num_checkpoints

        for i in range(num_total_samples):

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

                if (i - num_samples_to_discard + 1) % num_samples_checkpoint == 0:
                    # sanity check
                    print(
                        f"Num of collected samples so far is {self.part_obs_count_vec.sum()}")

                    # the -1 is because it should represent the index
                    checkpoint_num = ((i - num_samples_to_discard + 1) // num_samples_checkpoint) - 1
                    #
                    # for first_action in range(self.num_actions):
                    #     for second_action in range(self.num_actions):
                    #         action_pair_results = self.part_obs_count_vec.sum(axis=3)
                    #         action_pair_results = action_pair_results.sum(axis=2)
                    #         print(f"Number of times for action pair {first_action}-{second_action} is {action_pair_results}")

                    self.estimate_transition_matrix(
                        checkpoint_num=checkpoint_num)
                    # self.policy.update_transition_matrix(
                    #     estimated_transition_matrix=estimated_trans_matrix)

            first_state = next_state

        return self.estimated_action_state_dist, self.estimated_transition_matrix, self.error_frobenious_norm


    def estimate_transition_matrix(self, checkpoint_num):

        count_vector = self.part_obs_count_vec.copy()

        # estimate d(a,a',o,o') from count vector
        count_vector = count_vector.reshape(-1)
        act_obs_pair_dist = count_vector / count_vector.sum()

        # estimated d(a,a',s,s')
        sliced_act_stat_pair_dist = np.zeros(shape=(self.num_actions**2 * self.num_states**2))

        for first_action in range(self.num_actions):
            for second_action in range(self.num_actions):
                row_index = (first_action * self.num_actions + second_action) * self.num_obs**2
                col_index = (first_action * self.num_actions + second_action) * self.num_states**2

                act_obs_index = (first_action * self.num_actions + second_action) * self.num_obs**2
                current_act_obs_pair_dist = act_obs_pair_dist[act_obs_index:act_obs_index+self.num_obs**2]
                current_reference_matrix = self.pomdp.reference_matrix[
                                           row_index:row_index + self.num_obs ** 2,
                                           col_index:col_index + self.num_states ** 2]

                current_act_stat_pair_dist = np.linalg.lstsq(current_reference_matrix, current_act_obs_pair_dist,
                                rcond=None)[0]

                act_stat_index = (first_action * self.num_actions + second_action) * self.num_states ** 2
                sliced_act_stat_pair_dist[act_stat_index:act_stat_index+self.num_states**2] = current_act_stat_pair_dist

        # compute d(a,a',s,s')
        # act_stat_pair_dist = np.linalg.lstsq(self.pomdp.reference_matrix, act_obs_pair_dist, rcond=None)[0]

        act_stat_pair_dist = sliced_act_stat_pair_dist
        # print(f"Sliced version is {sliced_act_stat_pair_dist}")
        # print(f"Integral version is {act_stat_pair_dist}")

        act_stat_pair_dist = act_stat_pair_dist.reshape(
            (self.num_actions, self.num_actions, self.num_states, self.num_states))

        self.estimated_action_state_dist[checkpoint_num] = act_stat_pair_dist

        reduced_act_stat_pair_dist = act_stat_pair_dist.sum(axis=1)
        sum_over_last_state = reduced_act_stat_pair_dist.sum(axis=2)

        # this is expressed in the form action, state, state
        estimated_transition_matrix = reduced_act_stat_pair_dist / sum_over_last_state[:, :, None]
        reshaped_transition_matrix = np.empty(shape=(self.num_states, self.num_actions, self.num_states))
        for action in range(self.num_actions):
            reshaped_transition_matrix[:, action, :] = estimated_transition_matrix[action, :, :]

        self.estimated_transition_matrix[checkpoint_num] = reshaped_transition_matrix

        distance_matrix = np.absolute(self.pomdp.state_action_transition_matrix.reshape(-1) -
            reshaped_transition_matrix.reshape(-1))

        frobenious_norm = np.sqrt(np.sum(distance_matrix**2))
        self.error_frobenious_norm[checkpoint_num] = frobenious_norm

        probability_matrix_estimation_error = abs(np.sum(distance_matrix))
        print(f"Distance vector norm-1 is {probability_matrix_estimation_error}")
        print(
            f"Distance vector, frobenious norm, is {frobenious_norm}")

        print("Real transition matrix is")
        print(self.pomdp.state_action_transition_matrix)

        print("Estimated transition matrix is")
        print(reshaped_transition_matrix)


    # def old_estimate_transition_matrix(self, checkpoint_num):
    #
    #     count_vector = self.part_obs_count_vec.copy()
    #
    #     # estimate d(a,a',o,o') from count vector
    #     count_vector = count_vector.reshape(-1)
    #     act_obs_pair_dist = count_vector / count_vector.sum()
    #
    #     # compute d(a,a',s,s')
    #     act_stat_pair_dist = np.linalg.lstsq(self.pomdp.reference_matrix, act_obs_pair_dist, rcond=None)[0]
    #
    #     act_stat_pair_dist = act_stat_pair_dist.reshape(
    #         (self.num_actions, self.num_actions, self.num_states, self.num_states))
    #
    #     self.estimated_action_state_dist[checkpoint_num] = act_stat_pair_dist
    #
    #     reduced_act_stat_pair_dist = act_stat_pair_dist.sum(axis=1)
    #     sum_over_last_state = reduced_act_stat_pair_dist.sum(axis=2)
    #
    #     # this is expressed in the form action, state, state
    #     estimated_transition_matrix = reduced_act_stat_pair_dist / sum_over_last_state[:, :, None]
    #     reshaped_transition_matrix = np.empty(shape=(self.num_states, self.num_actions, self.num_states))
    #     for action in range(self.num_actions):
    #         reshaped_transition_matrix[:, action, :] = estimated_transition_matrix[action, :, :]
    #
    #     self.estimated_transition_matrix[checkpoint_num] = reshaped_transition_matrix
    #
    #     distance_matrix = np.absolute(self.pomdp.state_action_transition_matrix.reshape(-1) -
    #         reshaped_transition_matrix.reshape(-1))
    #
    #     frobenious_norm = np.sqrt(np.sum(distance_matrix**2))
    #     self.error_frobenious_norm[checkpoint_num] = frobenious_norm
    #
    #     probability_matrix_estimation_error = abs(np.sum(distance_matrix))
    #     print(f"Distance vector norm-1 is {probability_matrix_estimation_error}")
    #     print(
    #         f"Distance vector, frobenious norm, is {frobenious_norm}")
    #
    #     print("Real transition matrix is")
    #     print(self.pomdp.state_action_transition_matrix)
    #
    #     print("Estimated transition matrix is")
    #     print(reshaped_transition_matrix)