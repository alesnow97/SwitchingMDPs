import numpy as np

from strategy.tensor_utils import tensor_prod, find_best_permutation, \
    whitening, rtp


class SpectralHelper:

    def __init__(self,
                 num_states,
                 num_obs,
                 action_index,
                 action_transition_matrix: np.ndarray,
                 action_observation_matrix: np.ndarray,
                 possible_rewards,
                 tau_1,
                 num_episodes,
                 ):
        self.num_states = num_states
        self.num_obs = num_obs
        self.action_index = action_index
        self.possible_rewards = possible_rewards
        self.action_transition_matrix = action_transition_matrix
        self.action_observation_matrix = action_observation_matrix
        self.tau_1 = int(tau_1 // 3)
        self.num_episodes = num_episodes
        self.min_transition_prob = self.action_transition_matrix.min()
        self.min_observation_prob = self.action_observation_matrix.min()
        self.general_obs_1 = np.zeros((self.num_obs, self.tau_1*self.num_episodes))
        self.general_obs_2 = np.zeros((self.num_obs, self.tau_1*self.num_episodes))
        self.general_obs_3 = np.zeros((self.num_obs, self.tau_1*self.num_episodes))

    def run(self, current_state, episode_num):

        # experiments_samples = [n_samples // 3 for n_samples in experiments_samples]
        # total_horizon = experiments_samples[-1]

        collected_samples = np.zeros(shape=(self.tau_1*3, 3))

        n_models = self.num_states
        # n_distributions = self.switching_env.num_actions
        distr_dim = self.num_obs
        obs_dim = distr_dim
        low_dim = self.num_states
        n_tasks = self.tau_1
        use_true_means = True
        use_svd = True

        # sparse transition matrix
        T = self.action_transition_matrix.T
        O = self.action_observation_matrix.T

        # initial state distribution
        pi = np.ones(n_models) / n_models

        # random observation conditional means (assume probabilities of a multinomial distribution)
        # O = np.empty((obs_dim, n_models))
        # for state in range(n_models):
        #     curr_state_ac_obs = self.switching_env.state_action_reward_matrix[state, :, :].reshape(-1)
        #     O[:, state] = curr_state_ac_obs

        print("Initial state distribution")
        print(pi)
        print()
        print("Transition matrix")
        print(T)
        print()
        print("Observation means")
        print(O)
        print()

        # matrices to contain the generated observations
        # general_obs_1 = np.zeros((obs_dim, n_tasks))
        # general_obs_2 = np.zeros((obs_dim, n_tasks))
        # general_obs_3 = np.zeros((obs_dim, n_tasks))

        # Generate tasks/observations
        for j in range(self.tau_1*episode_num, self.tau_1*(episode_num+1)):

            # generate next three tasks
            if j % self.tau_1 * episode_num == 0:
                theta_1 = current_state
            else:
                theta_1 = np.random.choice(n_models, p=T[:,
                                                       theta_3])  # random next task

            theta_2 = np.random.choice(n_models, p=T[:, theta_1])
            theta_3 = np.random.choice(n_models, p=T[:, theta_2])

            obs1 = O[:, theta_1] / O[:, theta_1].sum()
            o1 = np.random.choice(obs_dim, size=1, p=obs1)[0]

            obs2 = O[:, theta_2] / O[:, theta_2].sum()
            o2 = np.random.choice(obs_dim, size=1, p=obs2)[0]

            obs3 = O[:, theta_3] / O[:, theta_3].sum()
            o3 = np.random.choice(obs_dim, size=1, p=obs3)[0]

            # get three observations
            if use_true_means:
                self.general_obs_1[:, j] = O[:, theta_1]
                self.general_obs_2[:, j] = O[:, theta_2]
                self.general_obs_3[:, j] = O[:, theta_3]
            else:
                # obs1 = O[:, theta_1] / O[:, theta_1].sum()
                # o1 = np.random.choice(obs_dim, size=1, p=obs1)[0]
                self.general_obs_1[o1, j] = 1

                # obs2 = O[:, theta_2] / O[:, theta_2].sum()
                # o2 = np.random.choice(obs_dim, size=1, p=obs2)[0]
                self.general_obs_2[o2, j] = 1

                # obs3 = O[:, theta_3] / O[:, theta_3].sum()
                # o3 = np.random.choice(obs_dim, size=1, p=obs3)[0]
                self.general_obs_3[o3, j] = 1

            i = j - self.tau_1*episode_num
            collected_samples[3*i] = np.array([self.action_index, o1, self.possible_rewards[o1]])
            collected_samples[3*i+1] = np.array([self.action_index, o2, self.possible_rewards[o2]])
            collected_samples[3*i+2] = np.array([self.action_index, o3, self.possible_rewards[o3]])

            if j % 5000 == 0:
                print(j)

        num_samples = (episode_num+1) * self.tau_1

        obs_1 = self.general_obs_1[:, :num_samples]
        obs_2 = self.general_obs_2[:, :num_samples]
        obs_3 = self.general_obs_3[:, :num_samples]

        error_occurred = False
        if use_svd:
            try:
                obs = np.concatenate([obs_1, obs_2, obs_3], axis=1)

                # perturb with gaussian noise
                obs += np.random.randn(obs.shape[0], obs.shape[1]) * 0.1

                u, s, v = np.linalg.svd(obs.T, full_matrices=False)
                obs_r = np.matmul(u[:, :low_dim], np.diag(s[:low_dim]))
                print("SVD error:",
                      np.abs(obs.T - np.matmul(obs_r, v[:low_dim, :])).max())
                print()

                obs = obs_r.T

                obs_1 = obs[:, :num_samples]
                obs_2 = obs[:, num_samples:2 * num_samples]
                obs_3 = obs[:, 2 * num_samples:]
            except Exception as e:
                print(f"An error occurred {e}")
                error_occurred = True

        if not error_occurred:
            # estimate covariance matrices
            K_12 = 0
            K_21 = 0
            K_31 = 0
            K_32 = 0

            for j in range(num_samples):
                K_12 += np.outer(obs_1[:, j], obs_2[:, j])
                K_21 += np.outer(obs_2[:, j], obs_1[:, j])
                K_31 += np.outer(obs_3[:, j], obs_1[:, j])
                K_32 += np.outer(obs_3[:, j], obs_2[:, j])

            K_12 /= num_samples
            K_21 /= num_samples
            K_31 /= num_samples
            K_32 /= num_samples

            # transform observations
            obs_1_mod = np.matmul(K_32, np.matmul(np.linalg.pinv(K_12), obs_1))
            obs_2_mod = np.matmul(K_31, np.matmul(np.linalg.pinv(K_21), obs_2))

            # estimate second and third moments
            M3_est = 0
            M2_est = 0
            for j in range(num_samples):
                M2_est += np.outer(obs_1_mod[:, j], obs_2_mod[:, j])
                M3_est += tensor_prod(obs_1_mod[:, j], obs_2_mod[:, j],
                                      obs_3[:, j])

            M2_est /= num_samples
            M3_est /= num_samples

            M2, M3 = M2_est, M3_est

            second_error_occurred = False
            try:
                M3, W = whitening(M3, M2, n_models)
                evalues, evectors = rtp(M3, 100, 100, faster=True)
                evalues = np.real(evalues)
                evectors = np.real(evectors)

                w_hat = 1 / evalues ** 2
                mu_3_hat = np.real(
                    np.matmul(np.linalg.pinv(W.T), evectors) * evalues[np.newaxis,
                                                               :])

                O_hat = np.real(
                    np.matmul(K_21, np.matmul(np.linalg.pinv(K_31), mu_3_hat)))
                T_hat = np.real(np.matmul(np.linalg.pinv(O_hat), mu_3_hat))

                # T_hat = T_hat / T_hat.sum(axis=0)[None, :]

                if use_svd:
                    O_hat = np.matmul(O_hat.T, v[:low_dim, :]).T

                print("Estimated probabilities")
                print(w_hat)
                print("Estimated mean observations")
                print(O_hat)
                print("Estimated transition matrix")
                print(T_hat)
                print()
            except Exception as e:
                print(f"An error occurred {e}")
                T_hat = np.random.rand(self.num_states, self.num_states)
                O_hat = np.random.rand(self.num_obs, self.num_states)

        else:
            T_hat = np.random.rand(self.num_states, self.num_states)
            O_hat = np.random.rand(self.num_obs, self.num_states)

        T_hat = np.maximum(T_hat, self.min_transition_prob)
        T_hat = T_hat / T_hat.sum(axis=0)[None, :]

        O_hat = np.maximum(O_hat, self.min_transition_prob + 0.01)
        O_hat = O_hat / O_hat.sum(axis=0)[None, :]

        # for i in range(n_distributions):
        #     subset_O_hat = O_hat[i*distr_dim:(i+1)*distr_dim, :]
        #     norm_subsest = subset_O_hat / subset_O_hat.sum(axis=0)[None, :]
        #     O_hat[i*distr_dim:(i+1)*distr_dim, :] = norm_subsest

        # compute errors (columns might be shuffled)
        err_T = []
        err_O = []
        for i in range(n_models):
            min_err_O = np.inf
            min_err_T = np.inf
            for j in range(n_models):
                err = np.abs(O[:, i] - O_hat[:, j]).max()
                if err < min_err_O:
                    min_err_O = err
                err = np.abs(T[:, i] - T_hat[:, j]).max()
                if err < min_err_T:
                    min_err_T = err
            err_T.append(min_err_T)
            err_O.append(min_err_O)

        print("Estimation error of O: ", max(err_O))
        print("Estimation error of T: ", max(err_T))

        print("Transition error")
        min_t_permutation, min_t_error = find_best_permutation(T, T_hat)

        print("Observation error")
        min_o_permutation, min_o_error = find_best_permutation(O, O_hat)

        print(min_t_permutation)
        print(min_o_permutation)

        T_hat = T_hat[:, min_t_permutation]
        O_hat = O_hat[:, min_o_permutation]

        print("Ciao")

        return collected_samples, T_hat.T, O_hat.T, theta_3

        # result_dict['hmm_o_error'] = max(err_O)
        # result_dict['hmm_t_error'] = max(err_T)
        # result_dict['min_t_permutation'] = min_t_permutation
        # result_dict['min_o_permutation'] = min_o_permutation
        # result_dict['norm_1_t_error'] = min_t_error
        # result_dict['norm_1_o_error_norm'] = min_o_error / (n_distributions * n_models)
        # result_dict['T_hat'] = T_hat.tolist()
        # result_dict['O_hat'] = O_hat.tolist()
        #
        # result_list.append(result_dict)
