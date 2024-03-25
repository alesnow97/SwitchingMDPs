import numpy as np

from utils import compute_min_svd


class MDP:

    def __init__(self,
                 num_states,
                 num_actions,
                 num_observations,
                 state_action_transition_matrix,
                 state_action_observation_matrix,
                 possible_rewards,
                 transition_multiplier=0,
                 observation_multiplier=5):
        self.num_states = num_states
        self.num_actions = num_actions
        self.num_obs = num_observations
        self.possible_rewards = possible_rewards
        self.transition_multiplier = transition_multiplier
        self.observation_multiplier = observation_multiplier
        self.state_action_transition_matrix = state_action_transition_matrix
        self.state_action_observation_matrix = state_action_observation_matrix

        if self.state_action_transition_matrix is not None:
            self.state_action_transition_matrix = state_action_transition_matrix
            self.state_action_observation_matrix = state_action_observation_matrix
            # self.possible_rewards = possible_rewards
            # self.arm_emission_matrices = arm_feature_representation
        else:
            self.state_action_transition_matrix = self.generate_transition_matrix(transition_multiplier=transition_multiplier)
            self.state_action_observation_matrix = \
                self.generate_state_action_reward_dist(observation_multiplier=observation_multiplier)

    # def generate_memoryless_policy(self):
    #     memoryless_policy = np.random.random((self.num_obs, self.num_actions))
    #     memoryless_policy = memoryless_policy / memoryless_policy.sum(axis=1)[
    #                                             :, None]
    #     return memoryless_policy
    #
    # def generate_env(self):
    #     print(self.transition_matrix)
    #
    # def generate_transition_matrix(self, transition_multiplier):
    #     # by setting specific design we give more probability to self-loops
    #     diag_matrix = np.eye(self.num_states) * transition_multiplier
    #     markov_chain = np.random.random_integers(
    #         low=1, high=11, size=(self.num_states, self.num_states))
    #     markov_chain = markov_chain + diag_matrix
    #     transition_matrix = markov_chain / markov_chain.sum(axis=1)[:, None]
    #     return transition_matrix

    def generate_transition_matrix(self, transition_multiplier):
        # by setting specific design we give more probability to self-loops
        transition_matrix = None
        for state in range(self.num_states):
            state_actions_matrix = np.random.random((self.num_actions, self.num_states))
            state_actions_matrix = state_actions_matrix / state_actions_matrix.sum(axis=1)[:, None]
            if transition_matrix is None:
                transition_matrix = state_actions_matrix
            else:
                transition_matrix = np.concatenate([transition_matrix, state_actions_matrix], axis=0)
        print(transition_matrix)

        reshaped_transition_matrix = transition_matrix.reshape((self.num_states, self.num_actions, self.num_states))

        return reshaped_transition_matrix

    def generate_state_action_reward_dist(self, observation_multiplier):
        row_dim = self.num_states*self.num_actions
        state_action_reward_matrix = np.empty(
            shape=(row_dim, self.num_obs))
        perturbation_matrix = np.zeros(shape=(row_dim, self.num_obs))

        if row_dim >= self.num_obs:
            for i in range(row_dim // self.num_obs):
                perturbation_matrix[self.num_obs * i:self.num_obs * (i + 1),
                :] = observation_multiplier * np.eye(self.num_obs)
        else:
            for i in range(self.num_obs // row_dim):
                perturbation_matrix[:,
                row_dim * i:row_dim * (
                            i + 1)] = observation_multiplier * np.eye(
                    row_dim)

        for state in range(self.num_states):
            action_reward = np.random.random((self.num_actions, self.num_obs))
            state_action_reward_matrix[state*self.num_actions:(state+1)*self.num_actions] = action_reward

        if row_dim >= self.num_obs:
            permutation = np.random.permutation(row_dim)
            permuted_matrix = perturbation_matrix[permutation, :]
        else:
            permutation = np.random.permutation(self.num_obs)
            permuted_matrix = perturbation_matrix[:, permutation]

        state_action_reward_matrix += permuted_matrix
        state_action_reward_matrix = state_action_reward_matrix / state_action_reward_matrix.sum(axis=1)[:, None]

        min_svd = compute_min_svd(state_action_reward_matrix)
        print(f"min svd of state action reward matrix is {min_svd}")

        state_action_reward_tensor = state_action_reward_matrix.reshape((self.num_states, self.num_actions, self.num_obs))

        return state_action_reward_tensor

    # def generate_arm_emission_matrices(self):
    #     arms_representation = np.empty(
    #         shape=(self.num_actions, self.num_states, self.num_obs))
    #     for arm in range(self.num_actions):
    #         arms_representation[arm] = self.state_action_reward_matrix[:, arm,
    #                                    :]
    #
    #     for arm in range(self.num_actions):
    #         current_emission_matrix = arms_representation[arm]
    #         sigma_min = utils.compute_min_svd(current_emission_matrix)
    #         print(f"Sigma min is {sigma_min}")
    #
    #     return arms_representation
    #
    # def generate_arm_feature_matrices(self):
    #     arm_feature_dict = {}
    #
    #     for first_arm in range(self.num_actions):
    #         for second_arm in range(self.num_actions):
    #             first_arm_emission_matrix = self.arm_emission_matrices[
    #                 first_arm]
    #             second_arm_emission_matrix = self.arm_emission_matrices[
    #                 second_arm]
    #             kronecker_prod = np.kron(first_arm_emission_matrix,
    #                                      second_arm_emission_matrix)
    #             arm_feature_dict[(first_arm, second_arm)] = kronecker_prod
    #             print(first_arm_emission_matrix.shape)
    #             sigma_min_first_arm = utils.compute_min_svd(
    #                 first_arm_emission_matrix)
    #             sigma_min_second_arm = utils.compute_min_svd(
    #                 second_arm_emission_matrix)
    #             sigma_min_kronecker = utils.compute_min_svd(kronecker_prod)
    #
    #             print(f"Kronecker sigma min is {sigma_min_kronecker}")
    #             print(
    #                 f"Product sigma min is {sigma_min_first_arm * sigma_min_second_arm}")
    #
    #             if sigma_min_kronecker == sigma_min_first_arm * sigma_min_second_arm:
    #                 print("I should be printed")
    #
    #     return arm_feature_dict
    #
    # def compute_stationary_distribution(self):
    #     evals, evecs = np.linalg.eig(self.transition_matrix.T)
    #     evec1 = evecs[:, np.isclose(evals, 1)]
    #
    #     evec1 = evec1[:, 0]
    #     print(f"Stationary distribution before normalization is {evec1}")
    #
    #     stationary = evec1 / evec1.sum()
    #     print(f"Stationary distribution is {stationary}")
    #     print(stationary.real)
    #     return stationary.real
    #
    # def compute_transition_stationary_distribution(self):
    #     # real_transition_distribution = np.zeros(shape=(self.num_states, self.num_states))
    #     transition_stationary_distribution = self.stationary_distribution[:,
    #                                          None] * self.transition_matrix
    #     # for i in range(self.num_states):
    #     #     real_transition_distribution[i] = self.stationary_distribution[i] * \
    #     #                                       self.transition_matrix[i]
    #
    #     print(f"The transition matrix is \n{self.transition_matrix}")
    #     print(
    #         f"The stationary distribution is \n{self.stationary_distribution}")
    #     print(
    #         f"The transition stationary distribution is \n{transition_stationary_distribution}")
    #     return transition_stationary_distribution

    def generate_next_state(self, state, action):
        probs = self.state_action_transition_matrix[state, action].reshape(-1)
        next_state = np.random.multinomial(
            n=1, pvals=probs, size=1)[0].argmax()
        return next_state

    def generate_current_reward(self, state, action):
        probs = self.state_action_observation_matrix[state, action].reshape(-1)
        current_reward = np.random.multinomial(
            n=1, pvals=probs, size=1)[0].argmax()
        return current_reward

