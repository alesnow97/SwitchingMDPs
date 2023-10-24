from itertools import combinations
import numpy as np
from switchingBanditEnv import SwitchingBanditEnv


class OurPolicy:

    def __init__(self, switching_env: SwitchingBanditEnv, total_horizon, exploration_horizon,
                 num_selected_actions):
        self.real_transition_matrix = switching_env.transition_matrix
        self.real_transition_stationary_distribution = switching_env.transition_stationary_distribution
        self.num_combinations = switching_env.num_actions**2
        self.num_states = switching_env.num_states
        self.num_actions = switching_env.num_actions
        self.num_obs = switching_env.num_obs
        self.reference_matrix = switching_env.reference_matrix
        self.possible_rewards = switching_env.possible_rewards
        self.total_horizon = total_horizon

        self.compute_exploration_arms(num_selected_actions)
        self.num_exploration_combinations = self.num_exploration_actions**2

        self.exploration_horizon = self.compute_exploration_horizon(exploration_horizon)
        self.state_action_reward_matrix = switching_env.state_action_reward_matrix
        self.reset()

        # for testing purposes
        self.num_arms_combinations_pulls = np.zeros(shape=(self.num_actions, self.num_actions), dtype=int)

    def reset(self):
        self.belief = np.array([1 / self.num_states] * self.num_states)
        self.action_reward_list = []
        self.count_vector = np.zeros(shape=self.num_combinations*self.num_obs**2, dtype=int)
        self.expected_visit_vector = None
        self.transition_matrix = None
        self.t = 0

    def choose_arm(self):
        if self.t < self.exploration_horizon:
            combination = self.t % (self.num_exploration_combinations * 2)
            return self.flatten_combinations_of_pairs_of_arms[combination]
        if self.t == self.exploration_horizon:
            print(f"Matrix estimation with {self.num_exploration_actions} is starting at {self.t}")
            self.estimate_transition_matrix()
            self.compute_belief_from_start()
        current_state_arm_reward = self.state_action_reward_matrix
        scaled_matrix = self.belief[:, None, None] * current_state_arm_reward
        scaled_matrix = scaled_matrix * self.possible_rewards[None, None, :]
        reduced_scaled_matrix = scaled_matrix.sum(axis=2)
        chosen_action = np.argmax(reduced_scaled_matrix.sum(axis=0))
        return chosen_action

    def update(self, pulled_arm, observed_reward_index):
        if self.t < self.exploration_horizon:
            self.action_reward_list.append((pulled_arm, observed_reward_index))
        else:
            self.update_belief(pulled_arm, observed_reward_index)
        self.t += 1

    def estimate_transition_matrix(self):
        for k in range(int(self.exploration_horizon / 2)):
            first_arm, first_reward = self.action_reward_list[2*k]
            second_arm, second_reward = self.action_reward_list[2*k + 1]
            index = first_arm * self.num_actions * self.num_obs ** 2 + second_arm * self.num_obs ** 2 + first_reward * self.num_obs + second_reward
            self.count_vector[int(index)] += 1
            self.num_arms_combinations_pulls[first_arm, second_arm] += 1

        masked_count_vector = self.count_vector[self.selected_mask]
        reshaped_count_vector = masked_count_vector.reshape(-1, self.num_obs ** 2)
        sum_reshaped_count_vector = reshaped_count_vector.sum(axis=1)
        sum_over_obs = sum_reshaped_count_vector.reshape((1, -1))
        normalizing_vec = np.tile(sum_over_obs.transpose(),
                                  (1, self.num_obs ** 2)).reshape(-1)
        self.expected_visit_vector = masked_count_vector / normalizing_vec

        flatten_transition_distribution = np.linalg.lstsq(self.reference_matrix[self.selected_mask], self.expected_visit_vector, rcond=None)[0]
        flatten_transition_distribution = flatten_transition_distribution / flatten_transition_distribution.sum()

        transition_stationary_distribution = flatten_transition_distribution.reshape((self.num_states, self.num_states))
        self.transition_matrix = transition_stationary_distribution / \
                                 transition_stationary_distribution.sum(axis=1)[:, None]
        print(f"Estimated transition matrix is \n{self.transition_matrix}")
        print(f"Real transition matrix is \n{self.real_transition_matrix}")

        distance_matrix = np.absolute(self.transition_matrix.reshape(-1) -
            self.real_transition_matrix.reshape(-1))
        print(f"Distance vector is {abs(np.sum(distance_matrix))}")

    def compute_belief_from_start(self):
        for episode in self.action_reward_list:
            self.update_belief(episode[0], episode[1])

    def update_belief(self, pulled_arm, observerd_reward_index):
        observation_distribution = self.state_action_reward_matrix[:,
                                   pulled_arm, observerd_reward_index].reshape(-1)

        scaled_belief = self.belief * observation_distribution
        transitioned_belief = scaled_belief @ self.transition_matrix.T

        self.belief = transitioned_belief / transitioned_belief.sum()

    def compute_exploration_horizon(self, expl_horizon):
        one_sweep_arm_combinations_pull = self.num_exploration_combinations * 2
        num_sweeps = int(expl_horizon / one_sweep_arm_combinations_pull) + 1
        if expl_horizon % one_sweep_arm_combinations_pull == 0:
            return expl_horizon
        return one_sweep_arm_combinations_pull * num_sweeps

    def compute_exploration_arms(self, num_selected_actions):
        if (num_selected_actions >= self.num_actions) or \
                (num_selected_actions**2 * self.num_obs**2 < self.num_states**2):
            self.num_exploration_actions = self.num_actions
            self.selected_mask = np.array([i for i in range(self.num_actions ** 2 * self.num_obs ** 2)], dtype=int)
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
                min_singular_value = s.min()
                value = num_selected_actions ** 2 * np.sqrt(np.log(
                    2 * num_selected_actions ** 2 * self.num_obs ** 2)) / min_singular_value

                # subset_reference_matrices.append(subset_reference_matrix)
                masks.append(mask)
                value_to_minimize.append(value)

            # compute value of initial reference matrix
            u, s, vh = np.linalg.svd(self.reference_matrix, full_matrices=True)
            min_singular_value = s.min()
            reference_matrix_value = self.num_actions ** 2 * np.sqrt(
                np.log(2 * self.num_actions ** 2 * self.num_obs ** 2)) / min_singular_value

            if min(value_to_minimize) < reference_matrix_value:
                self.num_exploration_actions = num_selected_actions
                subset_actions_index = np.argmin(value_to_minimize)
                self.selected_mask = masks[subset_actions_index]
                exploration_actions = list(possible_action_subsets[subset_actions_index])
            else:
                self.num_exploration_actions = self.num_actions
                self.selected_mask = np.ones((self.num_actions ** 2 * self.num_obs ** 2), dtype=int)
                exploration_actions = [i for i in range(self.num_actions)]

        self.flatten_combinations_of_pairs_of_arms = np.zeros(shape=2 * self.num_exploration_actions ** 2,
                                                              dtype=int)
        for i, first_arm in enumerate(exploration_actions):
            for j, second_arm in enumerate(exploration_actions):
                index = 2 * (i * self.num_exploration_actions + j)
                self.flatten_combinations_of_pairs_of_arms[index:index + 2] = [first_arm,
                                                                               second_arm]








