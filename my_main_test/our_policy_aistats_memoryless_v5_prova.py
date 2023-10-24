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
        self.total_horizon = total_horizon

        self.action_obs_prob = np.random.random((self.num_actions, self.num_obs))
        self.state_action_probs = np.random.random((self.num_states, self.num_actions))
        self.couple_action_obs_prob = np.random.random((self.num_actions**2, self.num_obs**2))
        self.couple_state_action_probs = np.random.random((self.num_states**2, self.num_actions**2))

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
        self.state_action_reward_list = []
        self.expected_visit_vector = None
        self.transition_matrix = np.ones(shape=(self.num_states,self.num_states)) * 1/self.num_states

        self.V = np.zeros(shape=(self.num_states**2, self.num_states**2))
        self.c = np.zeros(shape=self.num_states**2)

        self.t = 0

    def choose_arm(self):
        # if self.t % 2 == 1 and self.t > 1:
        #     self.update_matrices()
        if self.t % 300000 == 1 and self.t > 1:
            self.compute_action_obs_equilibrium()
            # self.update_matrices()
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

    def update(self, current_state, pulled_arm, observed_reward_index):
        self.state_action_reward_list.append((current_state, pulled_arm, observed_reward_index))
        self.update_belief(pulled_arm, observed_reward_index)
        self.t += 1

    # def update_matrices(self):
    #     first_arm, first_reward = self.action_reward_list[-2]
    #     second_arm, second_reward = self.action_reward_list[-1]
    #     current_arm_representation = self.arm_feature_matrices[(first_arm, second_arm)]
    #
    #     # add here the update that multiplies the probabilities
    #     previous_obs = self.action_reward_list[-3][1]
    #
    #     # first_prob = self.memoryless_policy_for_update[previous_obs][first_arm]
    #     # second_prob = self.memoryless_policy_for_update[first_reward][second_arm]
    #
    #     # first_prob = self.memoryless_policy[previous_obs][first_arm]
    #     # second_prob = self.memoryless_policy[first_reward][second_arm]
    #     # probs = first_prob * second_prob
    #
    #     outer = np.dot(current_arm_representation, current_arm_representation.T)
    #     # self.V += outer / probs
    #     self.V += outer
    #
    #     # c_update
    #     observation_index = first_reward * self.num_obs + second_reward
    #     x_vector = np.zeros(shape=(self.num_obs**2))
    #     x_vector[observation_index] = 1
    #
    #     c_update = np.dot(current_arm_representation, x_vector)
    #
    #     # self.c += c_update / probs
    #     self.c += c_update

    def compute_action_obs_equilibrium(self):

        num_action_obs_occurrences = np.zeros(shape=(self.num_actions, self.num_obs))
        num_state_action_occurrences = np.zeros(shape=(self.num_states, self.num_actions))

        # compute for single step
        for i in range(len(self.state_action_reward_list)):
            state, action, obs = self.state_action_reward_list[i]
            num_action_obs_occurrences[action, obs] += 1
            num_state_action_occurrences[state, action] += 1

        new_action_obs_prob = num_action_obs_occurrences / num_action_obs_occurrences.sum()
        action_obs_distance = np.absolute(new_action_obs_prob.reshape(-1) -
                                      self.action_obs_prob.reshape(-1))
        print(f"Equilibrium action obs distance vector is {abs(np.sum(action_obs_distance))}")

        new_state_action_probs = num_state_action_occurrences / num_state_action_occurrences.sum()
        state_action_distance = np.absolute(new_state_action_probs.reshape(-1) -
                                      self.state_action_probs.reshape(-1))
        print(f"Equilibrium state action distance vector is {abs(np.sum(state_action_distance))}")

        self.action_obs_prob = new_action_obs_prob
        self.state_action_probs = new_state_action_probs


        num_couple_action_obs_occurrences = np.zeros(
            shape=(self.num_actions**2, self.num_obs**2))
        num_couple_state_action_occurrences = np.zeros(shape=(self.num_states**2, self.num_actions**2))


        # compute for double step
        for i in range(len(self.state_action_reward_list) // 2):
            first_state, first_action, first_obs = self.state_action_reward_list[2*i]
            second_state, second_action, second_obs = self.state_action_reward_list[2*i+1]
            state_index = first_state * self.num_states + second_state
            action_index = first_action * self.num_actions + second_action
            obs_index = first_obs * self.num_obs + second_obs
            num_couple_action_obs_occurrences[action_index, obs_index] += 1
            num_couple_state_action_occurrences[state_index, action_index] += 1

        new_couple_action_obs_prob = num_couple_action_obs_occurrences / num_couple_action_obs_occurrences.sum()
        couple_action_obs_distance = np.absolute(new_couple_action_obs_prob.reshape(-1) -
                                      self.couple_action_obs_prob.reshape(-1))
        print(f"Equilibrium action obs couple distance vector is {abs(np.sum(couple_action_obs_distance))}")

        new_couple_state_action_probs = num_couple_state_action_occurrences / num_couple_state_action_occurrences.sum()
        couple_state_action_distance = np.absolute(new_couple_state_action_probs.reshape(-1) -
                                      self.couple_state_action_probs.reshape(-1))
        print(f"Equilibrium state action couple distance vector is {abs(np.sum(couple_state_action_distance))}")

        self.couple_action_obs_prob = new_couple_action_obs_prob
        self.couple_state_action_probs = new_couple_state_action_probs

        # further try
        kronecker = np.kron(new_state_action_probs, new_state_action_probs)
        kronecker_distance = np.absolute(kronecker.reshape(-1) -
                                      self.couple_state_action_probs.reshape(-1))
        print(f"Equilibrium kronecker state action couple distance vector is {abs(np.sum(kronecker_distance))}")

        # self.action_reward_list = []

        # num_couple_action_obs_occurrences = np.zeros(
        #     shape=(self.num_actions**2, self.num_obs**2))
        #
        # # compute for double step
        # for i in range(len(self.action_reward_list) // 2):
        #     first_action, first_obs = \
        #     self.action_reward_list[2 * i]
        #     second_action, second_obs = \
        #     self.action_reward_list[2 * i + 1]
        #     action_index = first_action * self.num_actions + second_action
        #     obs_index = first_obs * self.num_obs + second_obs
        #     num_couple_action_obs_occurrences[action_index, obs_index] += 1
        #
        # new_couple_action_obs_prob = num_couple_action_obs_occurrences / num_couple_action_obs_occurrences.sum()
        # couple_action_obs_distance = np.absolute(
        #     new_couple_action_obs_prob.reshape(-1) -
        #     self.couple_action_obs_prob.reshape(-1))
        # print(
        #     f"Couple action obs distance vector is {abs(np.sum(couple_action_obs_distance))}")
        #
        # self.couple_action_obs_prob = new_couple_action_obs_prob
        # self.action_reward_list = []

    def update_matrices(self):

        modified_arm_features = {}

        for first_action in range(self.num_actions):
            for second_action in range(self.num_actions):
                basic_arm_feature = self.arm_feature_matrices[(first_action, second_action)]

                # memoryless policy o -> a
                action_distribution = self.memoryless_policy[:, second_action]
                repeated_action_dist = np.repeat(action_distribution, self.num_obs)
                modified_arm_feature = basic_arm_feature * repeated_action_dist[None, :]

                modified_arm_features[(first_action,second_action)] = modified_arm_feature


        # construct the elements
        reference_matrix = None

        for first_action in range(self.num_actions):
            for second_action in range(self.num_actions):
                current_block = modified_arm_features[(first_action, second_action)]
                current_block = current_block.T

                if reference_matrix is None:
                    reference_matrix = current_block
                else:
                    reference_matrix = np.concatenate([reference_matrix, current_block], axis=0)

        linear_couple_action_obs = self.couple_action_obs_prob.reshape(-1)

        # solve the problem A*w = n
        #inv_ref = np.linalg.inv(reference_matrix)
        w_vector = np.linalg.lstsq(reference_matrix, linear_couple_action_obs, rcond=None)[0]
        # w_vector = np.dot(inv_ref, linear_couple_action_obs)

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


