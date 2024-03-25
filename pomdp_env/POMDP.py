import numpy as np

class POMDP:

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
        else:
            self.state_action_transition_matrix = self.generate_transition_matrix(transition_multiplier=transition_multiplier)
            self.state_action_observation_matrix = \
                self.generate_state_action_reward_dist(observation_multiplier=observation_multiplier)

        # self.observation_state_matrix = self.compute_diagonal_observation_state_matrix()
        self.reference_matrix = self.compute_reference_matrix()
        # self.reference_matrix_original = self.compute_reference_matrix_original()

        print("Ciao")


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

        min_svd = self.compute_min_svd(state_action_reward_matrix)
        print(f"min svd of state action reward matrix is {min_svd}")

        state_action_reward_tensor = state_action_reward_matrix.reshape((self.num_states, self.num_actions, self.num_obs))

        return state_action_reward_tensor


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

    def compute_diagonal_observation_state_matrix(self):
        row_dim = self.num_actions * self.num_obs
        col_dim = self.num_actions * self.num_states
        observation_state_big_matrix = np.zeros(shape=(row_dim, col_dim))

        for action in range(self.num_actions):
            state_observation_matrix = self.state_action_observation_matrix[:, action, :]
            observation_state_matrix = state_observation_matrix.T

            # kronecker product
            # kron = np.kron(first_mode_observation_matrix.T, second_mode_observation_matrix.T)
            row_index = action * self.num_obs
            col_index = action * self.num_states
            observation_state_big_matrix[row_index:row_index+self.num_obs, col_index:col_index+self.num_states] = observation_state_matrix

        min_svd = self.compute_min_svd(observation_state_big_matrix)
        print(f"min svd of observation mode big matrix is {min_svd}")

        return observation_state_big_matrix


    # def compute_reference_matrix(self):
    #     reference_from_kron = np.kron(self.observation_state_matrix, self.observation_state_matrix)
    #
    #     return reference_from_kron
    #     # min_svd = compute_min_svd(reference_matrix)
    #     # print(f"min svd of reference matrix is {min_svd}")

    def compute_reference_matrix(self):
        row_dim = self.num_actions**2 * self.num_obs**2
        col_dim = self.num_actions**2 * self.num_states**2
        reference_matrix = np.zeros(shape=(row_dim, col_dim))

        for first_action in range(self.num_actions):
            for second_action in range(self.num_actions):
                first_obs_mat = self.state_action_observation_matrix[:, first_action, :]
                second_obs_mat = self.state_action_observation_matrix[:, second_action, :]

                kron = np.kron(first_obs_mat.T,
                               second_obs_mat.T)

                row_index = (first_action * self.num_actions + second_action) * self.num_obs**2
                col_index = (first_action * self.num_actions + second_action) * self.num_states**2

                reference_matrix[row_index:row_index + self.num_obs ** 2,
                col_index:col_index + self.num_states ** 2] = kron

        # min_svd = compute_min_svd(reference_matrix)
        # print(f"min svd of reference matrix is {min_svd}")

        return reference_matrix

    def generate_pomdp_dict(self):
        save_dict = {}
        save_dict["num_states"] = self.num_states
        save_dict["num_actions"] = self.num_actions
        save_dict["num_obs"] = self.num_obs

        save_dict["transition_multiplier"] = self.transition_multiplier
        save_dict["observation_multiplier"] = self.observation_multiplier

        save_dict["state_action_transition_matrix"] = self.state_action_transition_matrix.tolist()
        save_dict["state_action_observation_matrix"] = self.state_action_observation_matrix.tolist()
        save_dict["possible_rewards"] = self.possible_rewards.tolist()

        return save_dict
        # # Convert and write JSON object to file
        # with open("sample.json", "w") as outfile:
        #     json.dump(save_dict, outfile)

    def compute_min_svd(self, reference_matrix):
        _, s, _ = np.linalg.svd(reference_matrix, full_matrices=True)
        print(f"Dimension of s is {len(s)}")
        return min(s)

