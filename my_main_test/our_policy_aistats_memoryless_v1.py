import numpy as np
from switchingBanditEnv_aistats import SwitchingBanditEnvAistats


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
        self.memoryless_policy = memoryless_policy
        self.memoryless_policy_for_update = memoryless_policy / memoryless_policy.sum(axis=0)[None, :]
        self.total_horizon = total_horizon

        # self.memoryless_policy = np.array([
        #     [0.5, 0.0, 0.5, 0.0, 0.0],
        #     [0.0, 0.5, 0.0, 0.5, 0.0],
        #     [0.7, 0.0, 0.3, 0.0, 0.0],
        #     [0.0, 0.0, 0.0, 0.6, 0.4],
        #     [0.0, 0.0, 0.5, 0.5, 0.0],
        #     [0.0, 0.8, 0.0, 0.0, 0.2],
        # ])

        # self.memoryless_policy = np.array([
        #     [0.06400655607991636,0.0,0.9359934439200837,0.0,0.0],
        #     [0.6397162440684465,0.0,0.3602837559315535,0.0,0.0],
        #     [0.023709243023322446,0.0,0.9762907569766776,0.0,0.0],
        #     [1.0,0.0,0.0,0.0,0.0],
        #     [1.0,0.0,0.0,0.0,0.0],
        #     [0.8730780603193377,0.0,0.12692193968066234,0.0,0.0]]
        # )

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
        if self.t % 2 == 1 and self.t > 1:
            self.update_matrices()
        if self.t % 2000000 == 1 and self.t > 1:
            self.estimate_transition_matrix()

        # if self.t == 0:
        #     chosen_action = np.random.choice(np.arange(self.num_actions))
        #     return chosen_action
        #
        # last_observation_index = self.action_reward_list[-1][1]
        # # print(last_observation_index)
        # action_probabilities = self.memoryless_policy[last_observation_index]
        #
        # chosen_action = np.random.choice(
        #     np.arange(self.num_actions),
        #     p=action_probabilities)

        # current_state_arm_reward = self.state_action_reward_matrix
        # scaled_matrix = self.belief[:, None, None] * current_state_arm_reward
        # scaled_matrix = scaled_matrix * self.possible_rewards[None, None, :]
        # reduced_scaled_matrix = scaled_matrix.sum(axis=2)
        # chosen_action = np.argmax(reduced_scaled_matrix.sum(axis=0))
        chosen_action = np.random.choice(np.arange(self.num_actions), p=np.array([0.3, 0.2, 0.1, 0.2, 0.2]))
        return chosen_action

    def update(self, pulled_arm, observed_reward_index):
        self.action_reward_list.append((pulled_arm, observed_reward_index))
        self.update_belief(pulled_arm, observed_reward_index)
        self.t += 1

    def update_matrices(self):
        first_arm, first_reward = self.action_reward_list[-2]
        second_arm, second_reward = self.action_reward_list[-1]
        current_arm_representation = self.arm_feature_matrices[(first_arm, second_arm)]

        # add here the update that multiplies the probabilities
        previous_obs = self.action_reward_list[-3][1]

        # first_prob = self.memoryless_policy_for_update[previous_obs][first_arm]
        # second_prob = self.memoryless_policy_for_update[first_reward][second_arm]

        # first_prob = self.memoryless_policy[previous_obs][first_arm]
        # second_prob = self.memoryless_policy[first_reward][second_arm]
        # probs = first_prob * second_prob

        outer = np.dot(current_arm_representation, current_arm_representation.T)
        # self.V += outer / probs
        self.V += outer

        # c_update
        observation_index = first_reward * self.num_obs + second_reward
        x_vector = np.zeros(shape=(self.num_obs**2))
        x_vector[observation_index] = 1

        c_update = np.dot(current_arm_representation, x_vector)

        # self.c += c_update / probs
        self.c += c_update

    def estimate_transition_matrix(self):
        w_vector = np.dot(np.linalg.inv(self.V), self.c)
        print(w_vector)

        w_vector = w_vector / w_vector.sum()
        w_matrix = w_vector.reshape((self.num_states, self.num_states))
        self.transition_matrix = w_matrix / w_matrix.sum(axis=1)[:, None]

        if self.t % 2000000 == 1:
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


