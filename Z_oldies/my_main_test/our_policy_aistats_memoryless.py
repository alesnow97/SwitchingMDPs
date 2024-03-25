import numpy as np
from Z_oldies.switchingBanditEnv_aistats import SwitchingBanditEnvAistats


class OurPolicyAistatsMemoryless:

    def __init__(self,
                 switching_env: SwitchingBanditEnvAistats,
                 memoryless_policy: np.ndarray,
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
        self.memoryless_policy = np.array([
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
        ])

        self.initial_reset()


    def initial_reset(self):
        self.belief = np.array([1 / self.num_states] * self.num_states)
        self.action_reward_list = []
        self.expected_visit_vector = None
        # self.transition_matrix = np.ones(shape=(self.num_states,self.num_states)) * 1/self.num_states
        self.transition_matrix = self.real_transition_matrix

        self.V = np.zeros(shape=(self.num_states**2, self.num_states**2))
        self.c = np.zeros(shape=self.num_states**2)

        self.t = 0

    def choose_arm(self):
        # if self.t % 2 == 1 and self.t > 1:
        #     self.update_matrices()
        if self.t % 200000 == 1:
            print(self.t)
        if self.t % 500000 == 1 and self.t > 1:
            self.compute_memoryless_policy()
            self.update_matrices()
            self.estimate_transition_matrix()

        # last_observation_index = self.action_reward_list[-1][1]
        # # print(last_observation_index)
        # action_probabilities = self.memoryless_policy[last_observation_index]
        #
        # chosen_action = np.random.choice(
        #     np.arange(self.num_actions),
        #     p=action_probabilities)

        # version that pulls an action sampled based on the probability of it
        # being optimal

        current_state_arm_reward = self.state_action_reward_matrix
        scaled_matrix = self.belief[:, None, None] * current_state_arm_reward
        scaled_matrix = scaled_matrix * self.possible_rewards[None, None, :]
        reduced_scaled_matrix = scaled_matrix.sum(axis=2)

        action_values = reduced_scaled_matrix.sum(axis=0)
        action_probability = action_values / action_values.sum()
        # chosen_action = np.argmax(reduced_scaled_matrix.sum(axis=0))
        chosen_action = np.random.choice(np.arange(self.num_actions), p=np.array(action_probability))
        return chosen_action

    def compute_memoryless_policy(self):
        num_obs_action_occurrences = np.zeros(shape=(self.num_obs, self.num_actions))

        for i in range(len(self.action_reward_list)-1):
            obs = self.action_reward_list[i][1]
            action = self.action_reward_list[i+1][0]
            num_obs_action_occurrences[obs, action] += 1

        new_memoryless_policy = num_obs_action_occurrences / num_obs_action_occurrences.sum(axis=1)[:, None]

        memoryless_policy_distance = np.absolute(new_memoryless_policy.reshape(-1) -
                                      self.memoryless_policy.reshape(-1))
        print(f"Memoryless distance vector is {abs(np.sum(memoryless_policy_distance))}")

        self.memoryless_policy = new_memoryless_policy

        # reset of action reward list
        # self.action_reward_list = []


    def update(self, pulled_arm, observed_reward_index):
        self.action_reward_list.append((pulled_arm, observed_reward_index))
        self.update_belief(pulled_arm, observed_reward_index)
        self.t += 1

    def update_matrices(self):

        self.V = np.zeros(shape=(self.num_states**2, self.num_states**2))
        self.c = np.zeros(shape=self.num_states**2)

        list_len = len(self.action_reward_list)
        for k in range(1, (list_len // 2 + 1)):
            first_arm, first_reward = self.action_reward_list[2*k-1]
            second_arm, second_reward = self.action_reward_list[2*k]
            current_arm_representation = self.arm_feature_matrices[(first_arm, second_arm)]

            # add here the update that multiplies the probabilities
            previous_obs = self.action_reward_list[2*k-2][1]
            first_prob = self.memoryless_policy[previous_obs][first_arm]
            second_prob = self.memoryless_policy[first_reward][second_arm]
            probs = first_prob * second_prob

            outer = np.dot(current_arm_representation, current_arm_representation.T)
            self.V += outer / probs

            # c_update
            observation_index = first_reward * self.num_obs + second_reward
            x_vector = np.zeros(shape=(self.num_obs**2))
            x_vector[observation_index] = 1

            c_update = np.dot(current_arm_representation, x_vector)

            self.c += c_update / probs

    def estimate_transition_matrix(self):
        w_vector = np.dot(np.linalg.inv(self.V), self.c)
        print(w_vector)

        w_vector = w_vector / w_vector.sum()
        w_matrix = w_vector.reshape((self.num_states, self.num_states))
        self.transition_matrix = w_matrix / w_matrix.sum(axis=1)[:, None]

        self.transition_matrix = np.maximum(self.transition_matrix, 0.02)

        if self.t % 500000 == 1:
            print(f"Estimated transition matrix is \n{self.transition_matrix}")
            print(f"Real transition matrix is \n{self.real_transition_matrix}")
            distance_matrix = np.absolute(self.transition_matrix.reshape(-1) -
                self.real_transition_matrix.reshape(-1))
            print(f"Distance vector is {abs(np.sum(distance_matrix))}")

            distance_matrix = np.absolute(self.transition_matrix.reshape(-1) -
                self.real_transition_matrix.reshape(-1))
            print(f"Distance vector is {abs(np.sum(distance_matrix))}")


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


