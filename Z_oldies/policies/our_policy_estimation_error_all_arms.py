import numpy as np
from Z_oldies.switchingBanditEnv import SwitchingBanditEnv


class OurPolicyTweakedAllArms:

    def __init__(self, switching_env: SwitchingBanditEnv,
                 starting_checkpoint,
                 checkpoint_duration,
                 num_checkpoints):
        self.real_transition_matrix = switching_env.transition_matrix
        self.real_transition_stationary_distribution = switching_env.transition_stationary_distribution
        self.num_combinations = switching_env.num_actions**2
        self.num_states = switching_env.num_states
        self.num_actions = switching_env.num_actions
        self.num_obs = switching_env.num_obs
        self.reference_matrix = switching_env.reference_matrix
        self.starting_checkpoint = starting_checkpoint
        self.checkpoint_duration = checkpoint_duration
        self.num_checkpoints = num_checkpoints
        self.total_horizon = starting_checkpoint + num_checkpoints * checkpoint_duration
        self.estimation_errors = np.empty((self.num_checkpoints, 2))
        self.k = 0

        self.exploration_actions = None
        self.compute_arms_combinations()
        self.num_exploration_combinations = self.num_actions**2

        self.state_action_reward_matrix = switching_env.state_action_reward_matrix

        # for testing purposes
        self.num_arms_combinations_pulls = np.zeros(shape=(self.num_actions, self.num_actions), dtype=int)

    def reset(self):
        self.action_reward_list = []
        self.count_vector = np.zeros(shape=self.num_combinations*self.num_obs**2, dtype=int)
        self.expected_visit_vector = None
        self.transition_matrix = None
        self.t = 0
        self.k = 0

    def choose_arm(self):
        if self.t - self.starting_checkpoint >= 0 and \
            (self.t - self.starting_checkpoint) % self.checkpoint_duration == 0:
            print(f"Matrix estimation with {self.num_actions}"
                  f" at {self.t}")
            self.estimate_transition_matrix()

        combination = self.t % (self.num_exploration_combinations * 2)
        return self.flatten_combinations_of_pairs_of_arms[combination]

    def update(self, pulled_arm, observed_reward_index):
        self.action_reward_list.append((pulled_arm, observed_reward_index))
        self.t += 1

    def estimate_transition_matrix(self):
        for k in range(self.k, int(self.t / 2)):
            first_arm, first_reward = self.action_reward_list[2*k]
            second_arm, second_reward = self.action_reward_list[2*k + 1]
            index = first_arm * self.num_actions * self.num_obs ** 2 + second_arm * self.num_obs ** 2 + first_reward * self.num_obs + second_reward
            self.count_vector[int(index)] += 1
            self.num_arms_combinations_pulls[first_arm, second_arm] += 1

        self.k = int(self.t / 2)
        # masked_count_vector = self.count_vector[self.selected_mask]
        reshaped_count_vector = self.count_vector.reshape(-1, self.num_obs ** 2)
        sum_reshaped_count_vector = reshaped_count_vector.sum(axis=1)
        sum_over_obs = sum_reshaped_count_vector.reshape((1, -1))
        normalizing_vec = np.tile(sum_over_obs.transpose(),
                                  (1, self.num_obs ** 2)).reshape(-1)
        self.expected_visit_vector = self.count_vector / normalizing_vec

        flatten_transition_distribution = np.linalg.lstsq(self.reference_matrix, self.expected_visit_vector, rcond=None)[0]
        flatten_transition_distribution = flatten_transition_distribution / flatten_transition_distribution.sum()

        transition_stationary_distribution = flatten_transition_distribution.reshape((self.num_states, self.num_states))
        self.transition_matrix = transition_stationary_distribution / \
                                 transition_stationary_distribution.sum(axis=1)[:, None]
        print(f"Estimated transition matrix is \n{self.transition_matrix}")
        print(f"Real transition matrix is \n{self.real_transition_matrix}")

        distance_matrix = np.absolute(self.transition_matrix.reshape(-1) -
            self.real_transition_matrix.reshape(-1))
        probability_matrix_estimation_error = abs(np.sum(distance_matrix))
        print(f"Distance vector is {probability_matrix_estimation_error}")

        update_index = int((self.t - self.starting_checkpoint) / self.checkpoint_duration)
        self.estimation_errors[update_index] = [self.t, probability_matrix_estimation_error]

    def compute_arms_combinations(self):
        self.num_exploration_actions = self.num_actions
        self.selected_mask = np.array([i for i in range(self.num_actions ** 2 * self.num_obs ** 2)], dtype=int)
        self.flatten_combinations_of_pairs_of_arms = np.zeros(shape=2 * self.num_actions ** 2,
                                                              dtype=int)
        for first_arm in range(self.num_actions):
            for second_arm in range(self.num_actions):
                index = 2 * (first_arm * self.num_actions + second_arm)
                self.flatten_combinations_of_pairs_of_arms[index:index + 2] = [first_arm,
                                                                               second_arm]







