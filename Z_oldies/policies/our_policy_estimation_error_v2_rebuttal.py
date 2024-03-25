from itertools import combinations
import numpy as np
from Z_oldies.switchingBanditEnv import SwitchingBanditEnv


class OurPolicyTweakedV2Rebuttal:

    def __init__(self, switching_env: SwitchingBanditEnv,
                 experiments_samples,
                 num_selected_actions):
        self.real_transition_matrix = switching_env.transition_matrix
        self.real_transition_stationary_distribution = switching_env.transition_stationary_distribution
        self.num_combinations = switching_env.num_actions**2
        self.num_states = switching_env.num_states
        self.num_actions = switching_env.num_actions
        self.num_obs = switching_env.num_obs
        self.reference_matrix = switching_env.reference_matrix
        self.experiments_samples = experiments_samples
        self.num_checkpoints = len(experiments_samples)
        self.total_horizon = experiments_samples[-1]
        self.min_estimation_errors = np.empty(self.num_checkpoints)

        self.exploration_actions = None
        self.compute_exploration_arms(num_selected_actions)
        self.num_exploration_combinations = self.num_exploration_actions**2

        self.state_action_reward_matrix = switching_env.state_action_reward_matrix

        # for testing purposes
        self.num_arms_combinations_pulls = np.zeros(shape=(self.num_actions, self.num_actions), dtype=int)

        self.min_action_reward_list = []
        self.min_count_vector = np.zeros(shape=self.num_combinations*self.num_obs**2, dtype=int)
        self.expected_visit_vector = None
        self.min_transition_matrix = None
        self.t = 0
        self.k = 0

    def choose_arm(self):
        if self.t in self.experiments_samples:
            print(f"Matrix estimation with {self.num_exploration_actions}"
                  f" at {self.t}")
            self.estimate_transition_matrix()

        combination = self.t % (self.num_exploration_combinations * 2)
        return self.min_flatten_combinations_of_pairs_of_arms[combination]

    def update(self, min_pulled_arm, min_observed_reward_index):
        self.min_action_reward_list.append((min_pulled_arm, min_observed_reward_index))
        self.t += 1

    def estimate_transition_matrix(self):
        for k in range(self.k, int(self.t / 2)):
            min_first_arm, min_first_reward = self.min_action_reward_list[2 * k]
            min_second_arm, min_second_reward = self.min_action_reward_list[2 * k + 1]
            min_index = min_first_arm * self.num_actions * self.num_obs ** 2 + min_second_arm * self.num_obs ** 2 + min_first_reward * self.num_obs + min_second_reward
            self.min_count_vector[int(min_index)] += 1

        self.k = int(self.t / 2)
        masked_count_vector = self.min_count_vector[self.min_selected_mask]
        reshaped_count_vector = masked_count_vector.reshape(-1, self.num_obs ** 2)
        sum_reshaped_count_vector = reshaped_count_vector.sum(axis=1)
        sum_over_obs = sum_reshaped_count_vector.reshape((1, -1))
        normalizing_vec = np.tile(sum_over_obs.transpose(),
                                  (1, self.num_obs ** 2)).reshape(-1)
        self.expected_visit_vector = masked_count_vector / normalizing_vec

        flatten_transition_distribution = np.linalg.lstsq(self.reference_matrix[self.min_selected_mask], self.expected_visit_vector, rcond=None)[0]
        flatten_transition_distribution = flatten_transition_distribution / flatten_transition_distribution.sum()

        transition_stationary_distribution = flatten_transition_distribution.reshape((self.num_states, self.num_states))
        self.min_transition_matrix = transition_stationary_distribution / \
                                 transition_stationary_distribution.sum(axis=1)[:, None]
        print(f"Estimated min transition matrix is \n{self.min_transition_matrix}")
        print(f"Real transition matrix is \n{self.real_transition_matrix}")

        distance_matrix = np.absolute(self.min_transition_matrix.reshape(-1) -
            self.real_transition_matrix.reshape(-1))
        probability_matrix_estimation_error = abs(np.sum(distance_matrix))
        print(f"Distance vector is {probability_matrix_estimation_error}")

        update_index = self.experiments_samples.index(self.t)
        self.min_estimation_errors[update_index] = probability_matrix_estimation_error

    def compute_exploration_arms(self, num_selected_actions):
        if (num_selected_actions >= self.num_actions) or \
                (num_selected_actions**2 * self.num_obs**2 < self.num_states**2):
            self.num_exploration_actions = self.num_actions
            exploration_actions = [i for i in range(self.num_actions)]
        else:
            combinations_of_pairs_of_arms = []
            for first_arm in range(self.num_actions):
                for second_arm in range(self.num_actions):
                    combinations_of_pairs_of_arms.append((first_arm, second_arm))

            # create all possible subsets of dimension num_selected_actions
            # and define their associated reference matrix
            possible_action_subsets = list(combinations([i for i in range(self.num_actions)], num_selected_actions))

            # subset_reference_matrices = []
            masks = []
            value_to_minimize = []
            minimum_singular_values = []
            for subset in possible_action_subsets:
                mask = np.ones((self.num_actions ** 2 * self.num_obs ** 2), dtype=int)
                for i, combination in enumerate(combinations_of_pairs_of_arms):
                    intersection = [value for value in combination if value not in subset]
                    if len(intersection) > 0:
                        mask[i * self.num_obs ** 2: (i + 1) * self.num_obs ** 2] = 0
                mask = np.where(mask)[0]
                subset_reference_matrix = self.reference_matrix[mask]
                print(subset_reference_matrix.shape)

                u, s, vh = np.linalg.svd(subset_reference_matrix,
                                         full_matrices=True)
                print(s)
                print(s.sum())
                min_singular_value = s.min()
                value = num_selected_actions ** 2 * np.sqrt(np.log(
                    2 * num_selected_actions ** 2 * self.num_obs ** 2)) / min_singular_value

                # subset_reference_matrices.append(subset_reference_matrix)
                masks.append(mask)
                value_to_minimize.append(value)
                minimum_singular_values.append(min_singular_value)

            min_subset_actions_index = np.argmin(value_to_minimize)
            max_subset_actions_index = np.argmax(value_to_minimize)

            self.min_min_singular_value = minimum_singular_values[min_subset_actions_index]
            self.max_min_singular_value = minimum_singular_values[max_subset_actions_index]
            self.min_selected_mask = masks[min_subset_actions_index]
            self.max_selected_mask = masks[max_subset_actions_index]

            self.min_exploration_actions = list(possible_action_subsets[min_subset_actions_index])
            self.max_exploration_actions = list(possible_action_subsets[max_subset_actions_index])

            self.min_complexity_value = min(value_to_minimize)
            self.max_complexity_value = max(value_to_minimize)



            #self.complexity_value = min(value_to_minimize)
            self.num_exploration_actions = num_selected_actions
            subset_actions_index = np.argmin(value_to_minimize)
            #self.min_min_singular_value = minimum_singular_values[subset_actions_index]
            #self.selected_mask = masks[subset_actions_index]
            #exploration_actions = list(possible_action_subsets[subset_actions_index])

        self.min_flatten_combinations_of_pairs_of_arms = np.zeros(shape=2 * self.num_exploration_actions ** 2,
                                                              dtype=int)
        self.max_flatten_combinations_of_pairs_of_arms = np.zeros(shape=2 * self.num_exploration_actions ** 2,
                                                              dtype=int)
        for i, first_arm in enumerate(self.min_exploration_actions):
            for j, second_arm in enumerate(self.min_exploration_actions):
                index = 2 * (i * self.num_exploration_actions + j)
                self.min_flatten_combinations_of_pairs_of_arms[index:index + 2] = [first_arm,
                                                                               second_arm]

        for i, first_arm in enumerate(self.max_exploration_actions):
            for j, second_arm in enumerate(self.max_exploration_actions):
                index = 2 * (i * self.num_exploration_actions + j)
                self.max_flatten_combinations_of_pairs_of_arms[index:index + 2] = [first_arm,
                                                                               second_arm]







