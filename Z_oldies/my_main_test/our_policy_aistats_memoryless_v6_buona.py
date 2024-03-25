import numpy as np

from Z_oldies import utils
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
        self.memoryless_policy = memoryless_policy
        self.total_horizon = total_horizon

        self.action_obs_prob = np.random.random((self.num_actions, self.num_obs))
        self.state_action_probs = np.random.random((self.num_states, self.num_actions))
        self.couple_action_obs_prob = np.random.random((self.num_actions**2, self.num_obs**2))
        self.couple_state_action_probs = np.random.random((self.num_states**2, self.num_actions**2))

        self.reference_matrix = None
        self.action_obs_obs_counter = None

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

        # this can be done in advance only because here we are using a
        # memoryless policy
        self.compute_equilibrium_formulation()

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
        # if self.t % 2 == 1 and self.t > 1:
        #     self.update_matrices()
        if self.t % 500000 == 1 and self.t > 1:
            self.update_count_vector()
            self.solve_equation()
            # self.estimate_transition_matrix()

        if self.t == 0:
            chosen_action = np.random.choice(np.arange(self.num_actions))
            return chosen_action

        last_observation_index = self.action_reward_list[-1][1]
        # print(last_observation_index)
        action_probabilities = self.memoryless_policy[last_observation_index]

        chosen_action = np.random.choice(
            np.arange(self.num_actions),
            p=action_probabilities)

        # current_state_arm_reward = self.state_action_reward_matrix
        # scaled_matrix = self.belief[:, None, None] * current_state_arm_reward
        # scaled_matrix = scaled_matrix * self.possible_rewards[None, None, :]
        # reduced_scaled_matrix = scaled_matrix.sum(axis=2)
        # chosen_action = np.argmax(reduced_scaled_matrix.sum(axis=0))
        # chosen_action = np.random.choice(np.arange(self.num_actions), p=np.array([0.3, 0.2, 0.1, 0.2, 0.2]))
        return chosen_action

    def update(self, pulled_arm, observed_reward_index):
        self.action_reward_list.append((pulled_arm, observed_reward_index))
        self.update_belief(pulled_arm, observed_reward_index)
        self.t += 1

    def compute_equilibrium_formulation(self):
        # we have S^2*A unknowns with O^2A equations
        reference_matrix = np.zeros(shape=(self.num_actions*self.num_obs**2,
                                           self.num_actions*self.num_states**2))

        # we consider all the three elements a_0, o_0, o_1
        for action in range(self.num_actions):
            starting_row_index = action * self.num_obs**2
            starting_col_index = action * self.num_states**2
            current_block_matrix = np.empty(shape=(self.num_obs**2, self.num_states**2))
            for first_state in range(self.num_states):
                for second_state in range(self.num_states):
                    for first_obs in range(self.num_obs):
                        for second_obs in range(self.num_obs):
                            obs_index_row = first_obs * self.num_obs + second_obs
                            state_index_col = first_state * self.num_states + second_state

                            p_o_given_second_state = self.state_action_reward_matrix[second_state, :, second_obs]
                            p_a_given_first_obs = self.memoryless_policy[first_obs]

                            dot_prod = np.dot(p_o_given_second_state, p_a_given_first_obs)
                            value = self.state_action_reward_matrix[first_state, action, first_obs] * dot_prod

                            current_block_matrix[obs_index_row, state_index_col] = value

            reference_matrix[
            starting_row_index:starting_row_index+self.num_obs**2,
            starting_col_index:starting_col_index+self.num_states**2] = (
                current_block_matrix)

        self.reference_matrix = reference_matrix
        sigma_min = utils.compute_min_svd(reference_matrix=self.reference_matrix)
        print(f"Sigma min is {sigma_min}")

        # compute info for other equations

        # store the reference matrix for solving equations with A*S unknowns
        action_obs_reference = np.zeros(shape=(self.num_actions*self.num_obs, self.num_actions*self.num_states))

        for action in range(self.num_actions):
            current_emission = self.arm_emission_matrices[action]
            transposed_emission = current_emission.T
            action_index_row = action * self.num_obs
            action_index_col = action * self.num_states
            action_obs_reference[
            action_index_row:action_index_row+self.num_obs,
            action_index_col:action_index_col+self.num_states] = (
                transposed_emission)

        self.action_obs_reference = action_obs_reference
        action_obs_sigma_min = utils.compute_min_svd(reference_matrix=self.action_obs_reference)
        print(f"Sigma min is {action_obs_sigma_min}")


    def update_count_vector(self):

        # update count of action obs obs vector
        action_obs_obs_counter = np.zeros(shape=(self.num_actions, self.num_obs, self.num_obs))
        for i in range(len(self.action_reward_list) - 1):
            action, first_obs = self.action_reward_list[i]
            second_obs = self.action_reward_list[i+1][1]
            action_obs_obs_counter[action, first_obs, second_obs] += 1

        self.action_obs_obs_counter = action_obs_obs_counter


        # update count of action obs vector
        # qui importa considerare o meno l'ultimo valore ??

        action_obs_counter = np.zeros(shape=(self.num_actions, self.num_obs))
        for i in range(len(self.action_reward_list)):
            action, obs = self.action_reward_list[i]
            action_obs_counter[action, obs] += 1

        self.action_obs_counter = action_obs_counter

    def solve_equation(self):

        # solve equation for action obs obs vector
        action_obs_obs_probs = self.action_obs_obs_counter.reshape(-1)
        action_obs_obs_probs = action_obs_obs_probs / action_obs_obs_probs.sum()
        unknowns = np.linalg.lstsq(self.reference_matrix, action_obs_obs_probs, rcond=None)[0]
        # w_vector = np.dot(inv_ref, linear_couple_action_obs)
        unknowns = unknowns.reshape((self.num_actions, self.num_states, self.num_states))
        print(unknowns)

        # solve equation for action obs vector
        action_obs_probs = self.action_obs_counter.reshape(-1)
        action_obs_probs = action_obs_probs / action_obs_probs.sum()
        action_obs_unknown = np.linalg.lstsq(self.action_obs_reference, action_obs_probs, rcond=None)[0]
        action_obs_unknown = action_obs_unknown.reshape((self.num_actions, self.num_states))
        print(action_obs_unknown)

        # for the moment just use one action for solving the problem
        chosen_action = 0
        chosen_action_obs_obs_unknowns = unknowns[chosen_action]
        chosen_action_obs_unknown = action_obs_unknown[chosen_action]
        estimated_transition_matrix = np.empty(shape=(self.num_states, self.num_states))
        for first_state in range(self.num_states):
            denominator = chosen_action_obs_unknown[first_state]
            for second_state in range(self.num_states):
                numerator = chosen_action_obs_obs_unknowns[first_state, second_state]
                estimated_transition_matrix[first_state, second_state] = numerator / denominator

        # normalize transition matrix
        estimated_transition_matrix = estimated_transition_matrix / estimated_transition_matrix.sum(axis=1)[:, None]
        self.transition_matrix = estimated_transition_matrix

        if self.t % 500000 == 1:
            print(f"Estimated transition matrix is \n{self.transition_matrix}")
            print(f"Real transition matrix is \n{self.real_transition_matrix}")
            distance_matrix = np.absolute(self.transition_matrix.reshape(-1) -
                self.real_transition_matrix.reshape(-1))
            print(f"Distance vector is {abs(np.sum(distance_matrix))}")






        # w_vector = w_vector / w_vector.sum()
        # w_matrix = w_vector.reshape((self.num_states, self.num_states))
        # self.transition_matrix = w_matrix / w_matrix.sum(axis=1)[:, None]
        #
        # self.transition_matrix = np.maximum(self.transition_matrix, 0.02)



    # def update_count_vector(self):
    #
    #     modified_arm_features = {}
    #
    #     for first_action in range(self.num_actions):
    #         for second_action in range(self.num_actions):
    #             basic_arm_feature = self.arm_feature_matrices[(first_action, second_action)]
    #
    #             # memoryless policy o -> a
    #             action_distribution = self.memoryless_policy[:, second_action]
    #             repeated_action_dist = np.repeat(action_distribution, self.num_obs)
    #             modified_arm_feature = basic_arm_feature * repeated_action_dist[None, :]
    #
    #             modified_arm_features[(first_action,second_action)] = modified_arm_feature
    #
    #
    #     # construct the elements
    #     reference_matrix = None
    #
    #     for first_action in range(self.num_actions):
    #         for second_action in range(self.num_actions):
    #             current_block = modified_arm_features[(first_action, second_action)]
    #             current_block = current_block.T
    #
    #             if reference_matrix is None:
    #                 reference_matrix = current_block
    #             else:
    #                 reference_matrix = np.concatenate([reference_matrix, current_block], axis=0)
    #
    #     linear_couple_action_obs = self.couple_action_obs_prob.reshape(-1)
    #
    #     # solve the problem A*w = n
    #     #inv_ref = np.linalg.inv(reference_matrix)
    #     w_vector = np.linalg.lstsq(reference_matrix, linear_couple_action_obs, rcond=None)[0]
    #     # w_vector = np.dot(inv_ref, linear_couple_action_obs)
    #
    #     w_vector = w_vector / w_vector.sum()
    #     w_matrix = w_vector.reshape((self.num_states, self.num_states))
    #     self.transition_matrix = w_matrix / w_matrix.sum(axis=1)[:, None]
    #
    #     self.transition_matrix = np.maximum(self.transition_matrix, 0.02)
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


    def estimate_transition_matrix(self):
        w_vector = np.dot(np.linalg.inv(self.V), self.c)
        print(w_vector)

        w_vector = w_vector / w_vector.sum()
        w_matrix = w_vector.reshape((self.num_states, self.num_states))
        self.transition_matrix = w_matrix / w_matrix.sum(axis=1)[:, None]

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


