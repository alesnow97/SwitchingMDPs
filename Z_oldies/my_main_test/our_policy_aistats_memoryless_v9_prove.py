import numpy as np

from Z_oldies import utils
from Z_oldies.switchingBanditEnv_aistats import SwitchingBanditEnvAistats


class OurPolicyAistatsMemoryless:

    def __init__(self,
                 switching_env: SwitchingBanditEnvAistats,
                 memoryless_policy: np.ndarray,
                 total_horizon):
        self.switching_env = switching_env
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
        self.total_horizon = total_horizon

        self.reference_matrix = None
        self.action_obs_obs_counter = None

        self.unknowns = np.random.random(size=self.num_actions**2*self.num_states**2)
        self.unknowns = self.unknowns / self.unknowns.sum()

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

        self.compute_equilibrium_formulation()

        self.initial_reset()


    def initial_reset(self):
        self.belief = np.array([1 / self.num_states] * self.num_states)
        self.action_reward_list = []
        self.expected_visit_vector = None
        self.transition_matrix = np.ones(shape=(self.num_states, self.num_states)) / self.num_states

        self.V = np.zeros(shape=(self.num_states**2, self.num_states**2))
        self.c = np.zeros(shape=self.num_states**2)

        self.t = 0

    def choose_arm(self):
        # if self.t % 2 == 1 and self.t > 1:
        #     self.update_matrices()
        if self.t % 500000 == 0 and self.t > 0:
            # self.compute_memoryless_policy()
            self.update_count_vector()
            self.solve_equation()
            # self.estimate_transition_matrix()

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

        current_state_arm_reward = self.state_action_reward_matrix
        scaled_matrix = self.belief[:, None, None] * current_state_arm_reward
        scaled_matrix = scaled_matrix * self.possible_rewards[None, None, :]
        reduced_scaled_matrix = scaled_matrix.sum(axis=2)
        chosen_action = np.argmax(reduced_scaled_matrix.sum(axis=0))
        # chosen_action = np.random.choice(np.arange(self.num_actions), p=np.array([0.3, 0.2, 0.1, 0.2, 0.2]))
        return chosen_action

    def update(self, pulled_arm, observed_reward_index):
        self.action_reward_list.append((pulled_arm, observed_reward_index))
        self.update_belief(pulled_arm, observed_reward_index)
        self.t += 1


    def compute_memoryless_policy(self):
        num_obs_action_occurrences = np.zeros(shape=(self.num_obs, self.num_actions))

        for i in range(len(self.action_reward_list)-1):
            obs = self.action_reward_list[i][1]
            action = self.action_reward_list[i+1][0]
            num_obs_action_occurrences[obs, action] += 1

        new_memoryless_policy = num_obs_action_occurrences / num_obs_action_occurrences.sum(axis=1)[:, None]

        memoryless_policy_distance = np.absolute(new_memoryless_policy.reshape(-1) -
                                      self.memoryless_policy.reshape(-1))
        print(f"Memoryless distance vector is {np.sum(memoryless_policy_distance)}")

        self.memoryless_policy = new_memoryless_policy

    def compute_equilibrium_formulation(self):
        # we have S^2*A^2 unknowns with O^2A^2 equations
        reference_matrix_row_dim = self.num_actions**2*self.num_obs**2
        reference_matrix_col_dim = self.num_actions**2*self.num_states**2
        reference_matrix = np.zeros(shape=(reference_matrix_row_dim,
                                           reference_matrix_col_dim))


        # we consider all the three elements a_0, a_0, o_0, o_1
        for first_action in range(self.num_actions):
            for second_action in range(self.num_actions):
                action_index = first_action * self.num_actions + second_action
                starting_row_index = action_index * self.num_obs**2
                starting_col_index = action_index * self.num_states**2
                first_action_emission_matrix = self.arm_emission_matrices[first_action]
                second_action_emission_matrix = self.arm_emission_matrices[second_action]
                kronecker_prod = np.kron(first_action_emission_matrix, second_action_emission_matrix)

                reference_matrix[
                starting_row_index:starting_row_index+self.num_obs ** 2,
                starting_col_index:starting_col_index + self.num_states ** 2,
                ] = kronecker_prod.T

        self.reference_matrix = reference_matrix
        sigma_min = utils.compute_min_svd(reference_matrix=self.reference_matrix)
        print(f"Sigma min is {sigma_min}")


    def update_count_vector(self):

        # update count of action obs couple vector
        action_obs_couple_counter = np.zeros(shape=(self.num_actions, self.num_actions, self.num_obs, self.num_obs))
        for i in range(len(self.action_reward_list) // 2):
            first_action, first_obs = self.action_reward_list[2*i]
            second_action, second_obs = self.action_reward_list[2*i+1]
            action_obs_couple_counter[first_action, second_action, first_obs, second_obs] += 1

        self.action_obs_couple_counter = action_obs_couple_counter

        self.action_reward_list = []


    def solve_equation(self):

        # solve equation for action obs obs vector
        action_obs_couple_counter = self.action_obs_couple_counter.reshape(-1)
        action_obs_couple_probs = action_obs_couple_counter / action_obs_couple_counter.sum()
        unknowns = np.linalg.lstsq(self.reference_matrix, action_obs_couple_probs, rcond=None)[0]
        # w_vector = np.dot(inv_ref, linear_couple_action_obs)
        unknowns = unknowns.reshape((self.num_actions, self.num_actions, self.num_states, self.num_states))
        print(unknowns)



        unknowns_over_first = unknowns.sum(axis=0)
        w_matrix = unknowns_over_first.sum(axis=0)
        print(w_matrix.shape)

        w_matrix = w_matrix / w_matrix.sum()
        # w_matrix = w_vector.reshape((self.num_states, self.num_states))
        self.transition_matrix = w_matrix / w_matrix.sum(axis=1)[:, None]

        # self.transition_matrix = np.maximum(self.transition_matrix, 0.02)

        if self.t % 500000 == 0:
            print(f"Estimated transition matrix is \n{self.transition_matrix}")
            print(f"Real transition matrix is \n{self.real_transition_matrix}")
            distance_matrix = np.absolute(self.transition_matrix.reshape(-1) -
                self.real_transition_matrix.reshape(-1))
            print(f"Distance vector is {abs(np.sum(distance_matrix))}")
            self.transition_matrix = self.switching_env.generate_transition_matrix_v2(
                transition_multiplier=20)

            unknowns_distance_matrix = np.absolute(unknowns.reshape(-1) -
                self.unknowns.reshape(-1))
            print(f"Unknown Distance vector is {abs(np.sum(unknowns_distance_matrix))}")
            self.unknowns = unknowns




    # def estimate_transition_matrix(self):
    #     w_vector = np.dot(np.linalg.inv(self.V), self.c)
    #     print(w_vector)
    #
    #     w_vector = w_vector / w_vector.sum()
    #     w_matrix = w_vector.reshape((self.num_states, self.num_states))
    #     self.transition_matrix = w_matrix / w_matrix.sum(axis=1)[:, None]
    #
    #     if self.t % 500000 == 1:
    #         print(f"Estimated transition matrix is \n{self.transition_matrix}")
    #         print(f"Real transition matrix is \n{self.real_transition_matrix}")
    #         distance_matrix = np.absolute(self.transition_matrix.reshape(-1) -
    #             self.real_transition_matrix.reshape(-1))
    #         print(f"Distance vector is {abs(np.sum(distance_matrix))}")
    #
    #         distance_matrix = np.absolute(self.transition_matrix.reshape(-1) -
    #             self.real_transition_matrix.reshape(-1))
    #         print(f"Distance vector is {abs(np.sum(distance_matrix))}")


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


