import numpy as np
from Z_oldies import utils


class SwitchingBanditEnvAistats:

    def __init__(self, num_states, num_actions, num_obs,
                 transition_matrix=None,
                 state_action_reward_matrix=None,
                 arm_feature_representation=None,
                 possible_rewards=None,
                 transition_multiplier=0,
                 observation_multiplier=5,
                 memoryless=False):
        self.num_states = num_states
        self.num_actions = num_actions
        self.num_obs = num_obs
        self.observation_multiplier = observation_multiplier
        self.transition_multiplier = transition_multiplier
        self.memoryless = memoryless

        if transition_matrix is not None:
            self.transition_matrix = transition_matrix
            self.state_action_reward_matrix = state_action_reward_matrix
            self.possible_rewards = possible_rewards
            self.arm_emission_matrices = arm_feature_representation
        else:
            self.transition_matrix = self.generate_transition_matrix_v2(transition_multiplier=transition_multiplier)
            self.state_action_reward_matrix = \
                self.generate_state_action_reward_dist(observation_multiplier=observation_multiplier)
            # with reference matrix we mean the A matrix
            self.arm_emission_matrices = self.generate_arm_emission_matrices()
            self.arm_feature_matrices = self.generate_arm_feature_matrices()
            self.possible_rewards = np.random.random_integers(low=1, high=20, size=self.num_obs)
        self.stationary_distribution = self.compute_stationary_distribution()
        self.transition_stationary_distribution = self.compute_transition_stationary_distribution()

        # self.sigma_min = utils.compute_min_svd(self.arm_emission_matrices)
        self.second_eigenvalue = utils.compute_second_eigenvalue(self.transition_matrix)
        # print(f"The smaller sigma is {self.sigma_min}")

        self.memoryless_policy = None
        if self.memoryless is True:
            self.memoryless_policy = self.generate_memoryless_policy()


    def generate_memoryless_policy(self):
        memoryless_policy = np.random.random((self.num_obs, self.num_actions))
        memoryless_policy = memoryless_policy / memoryless_policy.sum(axis=1)[:, None]
        return memoryless_policy

    def generate_env(self):
        print(self.transition_matrix)

    def generate_transition_matrix(self, transition_multiplier):
        # by setting specific design we give more probability to self-loops
        diag_matrix = np.eye(self.num_states) * transition_multiplier
        markov_chain = np.random.random_integers(
            low=1, high=11, size=(self.num_states, self.num_states))
        markov_chain = markov_chain + diag_matrix
        transition_matrix = markov_chain / markov_chain.sum(axis=1)[:, None]
        return transition_matrix

    def generate_transition_matrix_v2(self, transition_multiplier):
        # by setting specific design we give more probability to self-loops
        diag_matrix = np.eye(self.num_states) * transition_multiplier
        for state in range(self.num_states):
            outcome = np.random.random_integers(low=0, high=self.num_states, size=1)[0]
            while outcome != state:
                outcome = \
                np.random.random_integers(low=0, high=self.num_states, size=1)[
                    0]
            diag_matrix[state, outcome] += transition_multiplier * (2/3)
        markov_chain = np.random.random_integers(
            low=1, high=15, size=(self.num_states, self.num_states))
        markov_chain = markov_chain + diag_matrix
        transition_matrix = markov_chain / markov_chain.sum(axis=1)[:, None]
        return transition_matrix

    def generate_state_action_reward_dist(self, observation_multiplier):
        state_action_reward_matrix = np.empty(
            shape=(self.num_states, self.num_actions, self.num_obs))
        perturbation_matrix = np.zeros(shape=(self.num_actions, self.num_obs))

        if self.num_actions >= self.num_obs:
            for i in range(int(self.num_actions / self.num_obs)):
                perturbation_matrix[self.num_obs*i:self.num_obs*(i+1), :] = observation_multiplier * np.eye(self.num_obs)
        else:
            for i in range(int(self.num_obs / self.num_actions)):
                perturbation_matrix[:, self.num_actions*i:self.num_actions*(i+1)] = observation_multiplier * np.eye(self.num_actions)

        for state in range(self.num_states):
            action_reward = np.random.random((self.num_actions, self.num_obs))
            if self.num_actions >= self.num_obs:
                permutation = np.random.permutation(self.num_actions)
                permuted_matrix = perturbation_matrix[permutation, :]
            else:
                permutation = np.random.permutation(self.num_obs)
                permuted_matrix = perturbation_matrix[:, permutation]

            action_reward += permuted_matrix
            action_reward = action_reward / action_reward.sum(axis=1)[:, None]
            state_action_reward_matrix[state] = action_reward

            #for action in range(self.num_actions):
            #    categorical = np.random.random(size=self.num_obs)
            #    categorical = categorical / categorical.sum()
            #    state_action_reward_matrix[state, action, :] = \
            #        np.array(categorical)
        return state_action_reward_matrix

    def generate_arm_emission_matrices(self):
        arms_representation = np.empty(shape=(self.num_actions, self.num_states, self.num_obs))
        for arm in range(self.num_actions):
            current_emission_matrix = self.state_action_reward_matrix[:, arm, :]
            sigma_min = utils.compute_min_svd(current_emission_matrix)

            while sigma_min < 0.50:
                for state in range(self.num_states):
                    current_emission_matrix[state, np.random.choice(np.arange(self.num_obs))] += 2
                current_emission_matrix = current_emission_matrix / current_emission_matrix.sum(axis=1)[:, None]
                sigma_min = utils.compute_min_svd(current_emission_matrix)

            print(f"Sigma min is {sigma_min}")
            self.state_action_reward_matrix[:, arm, :] = current_emission_matrix
            arms_representation[arm] = current_emission_matrix

        return arms_representation

    def generate_arm_feature_matrices(self):
        arm_feature_dict = {}

        for first_arm in range(self.num_actions):
            for second_arm in range(self.num_actions):
                first_arm_emission_matrix = self.arm_emission_matrices[first_arm]
                second_arm_emission_matrix = self.arm_emission_matrices[second_arm]
                kronecker_prod = np.kron(first_arm_emission_matrix, second_arm_emission_matrix)
                arm_feature_dict[(first_arm, second_arm)] = kronecker_prod
                print(first_arm_emission_matrix.shape)
                sigma_min_first_arm = utils.compute_min_svd(first_arm_emission_matrix)
                sigma_min_second_arm = utils.compute_min_svd(second_arm_emission_matrix)
                sigma_min_kronecker = utils.compute_min_svd(kronecker_prod)

                print(f"Kronecker sigma min is {sigma_min_kronecker}")
                print(f"Product sigma min is {sigma_min_first_arm*sigma_min_second_arm}")

                if sigma_min_kronecker == sigma_min_first_arm * sigma_min_second_arm:
                    print("I should be printed")

        return arm_feature_dict

    def compute_stationary_distribution(self):
        evals, evecs = np.linalg.eig(self.transition_matrix.T)
        evec1 = evecs[:, np.isclose(evals, 1)]

        evec1 = evec1[:, 0]
        print(f"Stationary distribution before normalization is {evec1}")

        stationary = evec1 / evec1.sum()
        print(f"Stationary distribution is {stationary}")
        print(stationary.real)
        return stationary.real

    def compute_transition_stationary_distribution(self):
        # real_transition_distribution = np.zeros(shape=(self.num_states, self.num_states))
        transition_stationary_distribution = self.stationary_distribution[:, None] * self.transition_matrix
        # for i in range(self.num_states):
        #     real_transition_distribution[i] = self.stationary_distribution[i] * \
        #                                       self.transition_matrix[i]

        print(f"The transition matrix is \n{self.transition_matrix}")
        print(f"The stationary distribution is \n{self.stationary_distribution}")
        print(f"The transition stationary distribution is \n{transition_stationary_distribution}")
        return transition_stationary_distribution





