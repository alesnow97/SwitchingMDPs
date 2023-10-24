from itertools import combinations
import numpy as np
from switchingBanditEnv import SwitchingBanditEnv
from switchingBanditEnv_aistats import SwitchingBanditEnvAistats


class OurPolicyAistats:

    def __init__(self,
                 switching_env: SwitchingBanditEnvAistats,
                 total_horizon):
        self.real_transition_matrix = switching_env.transition_matrix
        self.real_transition_stationary_distribution = switching_env.transition_stationary_distribution
        self.num_states = switching_env.num_states
        self.num_actions = switching_env.num_actions
        self.num_obs = switching_env.num_obs
        self.arm_emission_matrices = switching_env.arm_emission_matrices
        self.arm_feature_matrices = switching_env.arm_feature_matrices
        self.possible_rewards = switching_env.possible_rewards
        self.state_action_reward_matrix = switching_env.state_action_reward_matrix
        self.total_horizon = total_horizon
        self.initial_reset()



    def initial_reset(self):
        self.belief = np.array([1 / self.num_states] * self.num_states)
        self.action_reward_list = []
        self.expected_visit_vector = None
        self.transition_matrix = np.ones(shape=(self.num_states,self.num_states)) * 1/self.num_states

        self.V = np.zeros(shape=(self.num_states**2, self.num_states**2))
        self.c = np.zeros(shape=self.num_states**2)

        self.t = 0

    def choose_arm(self):
        if self.t % 2 == 0 and self.t > 0:
            self.estimate_transition_matrix()
            self.compute_belief_from_start()

        # if self.t == self.exploration_horizon:
        #     print(f"Matrix estimation with {self.num_exploration_actions} is starting at {self.t}")
        #     self.estimate_transition_matrix()
        #     self.compute_belief_from_start()
        current_state_arm_reward = self.state_action_reward_matrix
        scaled_matrix = self.belief[:, None, None] * current_state_arm_reward
        scaled_matrix = scaled_matrix * self.possible_rewards[None, None, :]
        reduced_scaled_matrix = scaled_matrix.sum(axis=2)
        chosen_action = np.argmax(reduced_scaled_matrix.sum(axis=0))
        return chosen_action

    def update(self, pulled_arm, observed_reward_index):
        self.action_reward_list.append((pulled_arm, observed_reward_index))
        self.update_belief(pulled_arm, observed_reward_index)
        self.t += 1

    def estimate_transition_matrix(self):
        first_arm, first_reward = self.action_reward_list[-2]
        second_arm, second_reward = self.action_reward_list[-1]
        current_arm_representation = self.arm_feature_matrices[(first_arm, second_arm)]
        print(current_arm_representation.shape)

        outer = np.dot(current_arm_representation, current_arm_representation.T)
        self.V += outer

        # c_update
        observation_index = first_reward * self.num_obs + second_reward
        x_vector = np.zeros(shape=(self.num_obs**2))
        x_vector[observation_index] = 1

        c_update = np.dot(current_arm_representation, x_vector)
        self.c += c_update

        w_vector = np.dot(np.linalg.inv(self.V), self.c)
        print(w_vector)

        w_vector = w_vector / w_vector.sum()
        w_matrix = w_vector.reshape((self.num_states, self.num_states))
        self.transition_matrix = w_matrix / w_matrix.sum(axis=1)[:, None]

        if self.t % 10000 == 0:
            print(f"Estimated transition matrix is \n{self.transition_matrix}")
            print(f"Real transition matrix is \n{self.real_transition_matrix}")
            distance_matrix = np.absolute(self.transition_matrix.reshape(-1) -
                self.real_transition_matrix.reshape(-1))
            print(f"Distance vector is {abs(np.sum(distance_matrix))}")

            distance_matrix = np.absolute(self.transition_matrix.reshape(-1) -
                self.real_transition_matrix.reshape(-1))
            print(f"Distance vector is {abs(np.sum(distance_matrix))}")

        # for k in range(int(len(self.action_reward_list) / 2)):
        #     first_arm, first_reward = self.action_reward_list[2*k]
        #     second_arm, second_reward = self.action_reward_list[2*k + 1]
        #     index = first_arm * self.num_actions * self.num_obs ** 2 + second_arm * self.num_obs ** 2 + first_reward * self.num_obs + second_reward
        #     self.count_vector[int(index)] += 1
        #     self.num_arms_combinations_pulls[first_arm, second_arm] += 1
        #
        # masked_count_vector = self.count_vector[self.selected_mask]
        # reshaped_count_vector = masked_count_vector.reshape(-1, self.num_obs ** 2)
        # sum_reshaped_count_vector = reshaped_count_vector.sum(axis=1)
        # sum_over_obs = sum_reshaped_count_vector.reshape((1, -1))
        # normalizing_vec = np.tile(sum_over_obs.transpose(),
        #                           (1, self.num_obs ** 2)).reshape(-1)
        # self.expected_visit_vector = masked_count_vector / normalizing_vec
        #
        # flatten_transition_distribution = np.linalg.lstsq(self.reference_matrix[self.selected_mask], self.expected_visit_vector, rcond=None)[0]
        # flatten_transition_distribution = flatten_transition_distribution / flatten_transition_distribution.sum()
        #
        # transition_stationary_distribution = flatten_transition_distribution.reshape((self.num_states, self.num_states))
        # self.transition_matrix = transition_stationary_distribution / \
        #                          transition_stationary_distribution.sum(axis=1)[:, None]
        # print(f"Estimated transition matrix is \n{self.transition_matrix}")
        # print(f"Real transition matrix is \n{self.real_transition_matrix}")
        #
        # distance_matrix = np.absolute(self.transition_matrix.reshape(-1) -
        #     self.real_transition_matrix.reshape(-1))
        # print(f"Distance vector is {abs(np.sum(distance_matrix))}")

    def compute_belief_from_start(self):
        self.belief = np.array([1 / self.num_states] * self.num_states)
        for episode in self.action_reward_list:
            self.update_belief(episode[0], episode[1])

    def update_belief(self, pulled_arm, observed_reward_index):
        observation_distribution = self.state_action_reward_matrix[:,
                                   pulled_arm, observed_reward_index].reshape(-1)

        scaled_belief = self.belief * observation_distribution
        transitioned_belief = scaled_belief @ self.transition_matrix.T

        self.belief = transitioned_belief / transitioned_belief.sum()


