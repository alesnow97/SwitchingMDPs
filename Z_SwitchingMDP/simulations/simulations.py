import numpy as np

from Z_SwitchingMDP.environment.switchingMDPs import SwitchingMDPs
from policies.state_based_policy import StateBasedPolicy
from utils import compute_min_svd

np.random.seed(200)


class EstimationErrorSimulation:

    def __init__(self,
                 switching_mdps: SwitchingMDPs):
        self.switching_mdps = switching_mdps
        self.num_modes = self.switching_mdps.num_modes
        self.num_states = self.switching_mdps.num_states
        self.num_actions = self.switching_mdps.num_actions
        self.num_obs = self.switching_mdps.num_obs
        self.markov_chain_matrix = self.switching_mdps.markov_chain_matrix
        self.mode_state_action_observation_matrix = self.switching_mdps.mode_state_action_observation_matrix

        self.observation_mode_matrix = self.compute_observation_mode_matrix()
        self.reference_matrix = self.compute_reference_matrix()

        self.check_equivalence()


    def run(self, horizon: int, num_experiments: int):

        # bandit_info_dict = {
        #     'transition_matrix': self.switching_env.transition_matrix.tolist(),
        #     # 'reference_matrix': self.switching_env.arm.tolist(),
        #     'state_action_reward_matrix': self.switching_env.state_action_reward_matrix.tolist(),
        #     'possible_rewards': self.switching_env.possible_rewards.tolist(),
        #     'num_states': self.switching_env.num_states,
        #     'num_actions': self.switching_env.num_actions,
        #     'num_obs': self.switching_env.num_obs,
        #     'observation_multiplier': self.switching_env.observation_multiplier,
        #     'transition_multiplier': self.switching_env.transition_multiplier
        # }

        # result_dict = {'total_horizon': total_horizon,
        #                'epsilon': self.epsilon,
        #                'sliding_window_size': self.sliding_window_size,
        #                'exp3S_gamma': self.exp3S_gamma,
        #                'exp3S_normalization_factor': self.exp3S_normalization_factor,
        #                'exp3S_limit': self.exp3S_limit,
        #                'num_experiments': self.num_experiments,
        #                'algorithms_to_use': algorithms_to_use, 'rewards': {}}

        mode_state_action_rew_list = np.empty(
            shape=(num_experiments, horizon, 4), dtype=int)


        # state_action_reward_matrix = self.switching_env.state_action_reward_matrix
        # possible_rewards = self.switching_env.possible_rewards

        for n in range(num_experiments):
            print("experiment_n: " + str(n))
            current_mode = None
            current_state = None

            state_based_policy = StateBasedPolicy(num_states=self.num_states,
                                             num_actions=self.num_actions,
                                             num_obs=self.num_obs)

            count_vec = np.zeros(shape=(self.num_states, self.num_states,
                                        self.num_actions, self.num_actions,
                                        self.num_obs, self.num_obs))

            for i in range(horizon):

                if i % 2 == 0 and i > 0:
                    first_mode, first_state, first_action, first_obs = \
                    mode_state_action_rew_list[n, i-2]
                    second_mode, second_state, second_action, second_obs = \
                    mode_state_action_rew_list[n, i-1]
                    count_vec[
                        first_state, second_state, first_action,
                        second_action, first_obs, second_obs] += 1

                if i % 10000 == 0 and i > 0:
                    print(i)
                    if i % 100000 == 0:
                        self.estimate_all(count_vec)

                if i == 0:
                    current_mode = np.random.random_integers(
                        low=0, high=self.num_modes-1)
                    current_state = np.random.random_integers(
                        low=0, high=self.num_states-1)
                    print(f"First state is {current_state}")
                    print(f"First mode is {current_mode}")

                current_action = state_based_policy.choose_arm(current_state)
                next_state, current_rew = self.switching_mdps.get_next_state_reward(
                    mode=current_mode, state=current_state, action=current_action
                )

                mode_state_action_rew_list[n, i] = np.array([
                    current_mode, current_state, current_action, current_rew
                ])

                current_state = next_state
                current_mode = np.random.multinomial(
                    n=1, pvals=self.markov_chain_matrix[current_mode], size=1)[0].argmax()

        # if self.save_bandit_info:
        #     f = open(self.bandit_dir_path + '/bandit_info.json', 'w')
        #     json_file = json.dumps(bandit_info_dict)
        #     f.write(json_file)
        #     f.close()
        #
        # if self.save_results:
        #     f = open(self.exp_dir_path + '/exp_info.json', 'w')
        #     json_file = json.dumps(result_dict)
        #     f.write(json_file)
        #     f.close()

    def estimate_all(self, count_vector: np.ndarray):

        count_vector = count_vector.reshape(-1)
        probs = count_vector / count_vector.sum()

        computed_results = np.linalg.lstsq(self.reference_matrix, probs, rcond=None)[0]

        computed_results = computed_results.reshape(
            (self.num_states, self.num_states, self.num_actions, self.num_actions, self.num_modes, self.num_modes))

        # compute markov chain matrix
        w_matrix = computed_results.sum(axis=(0, 1, 2, 3))
        print(w_matrix)
        w_matrix = w_matrix / w_matrix.sum()
        # w_matrix = w_vector.reshape((self.num_states, self.num_states))
        markov_chain_matrix = w_matrix / w_matrix.sum(axis=1)[:, None]
        print(markov_chain_matrix)

        distance_matrix = np.absolute(self.markov_chain_matrix.reshape(-1) -
            markov_chain_matrix.reshape(-1))
        probability_matrix_estimation_error = abs(np.sum(distance_matrix))
        print(f"Distance vector is {probability_matrix_estimation_error}")

        for mode in range(self.num_modes):
            current_mode_results = computed_results[:, :, :, :, mode, mode]

            # marginalize over the second action
            current_mode_results = current_mode_results.sum(axis=3)
            reshaped_current_mode_results_1 = current_mode_results.swapaxes(1, 2)
            reshaped_current_mode_results_1 = reshaped_current_mode_results_1.reshape((self.num_states*self.num_actions, self.num_states))

            estimated_transition_matrix_1 = reshaped_current_mode_results_1 / reshaped_current_mode_results_1.sum(axis=1)[:, None]
            real_transition_matrix = self.switching_mdps.mode_state_action_transition_matrix[mode]
            real_transition_matrix = real_transition_matrix.reshape((self.num_states*self.num_actions, self.num_states))
            distance_matrix = np.absolute(
                estimated_transition_matrix_1.reshape(-1) -
                real_transition_matrix.reshape(-1))
            probability_matrix_estimation_error = abs(np.sum(distance_matrix))
            print(f"Distance vector for mode {mode} is {probability_matrix_estimation_error}")



    def compute_observation_mode_matrix(self):
        row_dim = self.num_states * self.num_actions * self.num_obs
        col_dim = self.num_states * self.num_actions * self.num_modes
        observation_mode_big_matrix = np.zeros(shape=(row_dim, col_dim))

        for state in range(self.num_states):
            for action in range(self.num_actions):
                mode_observation_matrix = self.mode_state_action_observation_matrix[:, state, action, :]
                observation_mode_matrix = mode_observation_matrix.T

                # kronecker product
                # kron = np.kron(first_mode_observation_matrix.T, second_mode_observation_matrix.T)
                row_index = state * self.num_actions * self.num_obs + \
                            action * self.num_obs
                col_index = state * self.num_actions * self.num_modes + \
                            action * self.num_modes
                observation_mode_big_matrix[row_index:row_index+self.num_obs, col_index:col_index+self.num_modes] = observation_mode_matrix

        min_svd = compute_min_svd(observation_mode_big_matrix)
        print(f"min svd of observation mode big matrix is {min_svd}")

        return observation_mode_big_matrix


    def compute_reference_matrix(self):
        row_dim = self.num_states**2 * self.num_actions**2 * self.num_obs**2
        col_dim = self.num_states**2 * self.num_actions**2 * self.num_modes**2
        reference_matrix = np.zeros(shape=(row_dim, col_dim))

        for first_state in range(self.num_states):
            for second_state in range(self.num_states):
                for first_action in range(self.num_actions):
                    for second_action in range(self.num_actions):
                        first_mode_observation_matrix = self.mode_state_action_observation_matrix[:, first_state, first_action, :]
                        second_mode_observation_matrix = self.mode_state_action_observation_matrix[:, second_state, second_action, :]

                        # kronecker product
                        kron = np.kron(first_mode_observation_matrix.T, second_mode_observation_matrix.T)
                        row_index = (first_state * self.num_states + second_state) * self.num_actions**2 * self.num_obs**2 + \
                                    (first_action * self.num_actions + second_action) * self.num_obs**2
                        col_index = (first_state * self.num_states + second_state) * self.num_actions**2 * self.num_modes**2 + \
                                    (first_action * self.num_actions + second_action) * self.num_modes**2
                        reference_matrix[row_index:row_index+self.num_obs**2, col_index:col_index+self.num_modes**2] = kron

        min_svd = compute_min_svd(reference_matrix)
        print(f"min svd of reference matrix is {min_svd}")

        return reference_matrix


    def check_equivalence(self):
        reference_from_kron = np.kron(self.observation_mode_matrix, self.observation_mode_matrix)

        min_svd_from_kron = compute_min_svd(reference_from_kron)
        print(f"min svd of reference matrix from Kronecker is {min_svd_from_kron}")

        min_svd = compute_min_svd(self.reference_matrix)
        print(f"min svd of reference matrix is {min_svd}")


