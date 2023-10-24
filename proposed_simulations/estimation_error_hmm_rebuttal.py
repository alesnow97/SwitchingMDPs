from policies.our_policy_estimation_error import OurPolicyTweaked
from switchingBanditEnv import SwitchingBanditEnv
from tensor_utils import tensor_prod, whitening, rtp
from utils import find_best_permutation
import numpy as np
import os
import json


class EstimationErrorExpRebuttal:

    def __init__(self, switching_env: SwitchingBanditEnv,
                 save_results, loaded_bandit, dir_name,
                 save_bandit_info=False, bandit_num=0, run_experiments=True):
        self.switching_env = switching_env
        self.bandit_dir_path = None
        self.new_bandit_index = None
        self.loaded_bandit = loaded_bandit
        self.bandit_num = bandit_num
        self.dir_name = dir_name
        self.generate_dirs()
        self.save_results = save_results
        self.run_experiments = run_experiments
        self.save_bandit_info = save_bandit_info

    def generate_dirs(self):
        dir_name = self.dir_name
        if os.path.exists(dir_name):
            if self.loaded_bandit:
                self.bandit_dir_path = dir_name + f'/bandit{self.bandit_num}'
                self.new_exp_index = len(os.listdir(self.bandit_dir_path)) - 1
            else:
                self.new_bandit_index = len(os.listdir(dir_name))
                self.bandit_dir_path = dir_name + f'/bandit{self.new_bandit_index}'
                os.mkdir(self.bandit_dir_path)
                self.new_exp_index = 0
        else:
            os.mkdir(dir_name)
            self.new_bandit_index = 0
            self.bandit_dir_path = dir_name + f'/bandit{self.new_bandit_index}'
            os.mkdir(self.bandit_dir_path)
            self.new_exp_index = 0
        self.exp_dir_path = self.bandit_dir_path + f'/exp_hmm_{self.new_exp_index}'
        os.mkdir(self.exp_dir_path)

    def run(self, experiments_samples):
        bandit_info_dict = {
            'transition_matrix': self.switching_env.transition_matrix.tolist(),
            'reference_matrix': self.switching_env.reference_matrix.tolist(),
            'state_action_reward_matrix': self.switching_env.state_action_reward_matrix.tolist(),
            'num_states': self.switching_env.num_states,
            'num_actions': self.switching_env.num_actions,
            'num_obs': self.switching_env.num_obs}

        # num_selected_arms_list = [1, 2, 3, 5]
        num_selected_arms_list = [2, 5, 10, 15]

        self.num_checkpoints = len(experiments_samples)
        experiments_samples = [n_samples // 3 for n_samples in experiments_samples]
        total_horizon = experiments_samples[-1]

        result_list = []

        if self.run_experiments:
            n_models = self.switching_env.num_states
            n_distributions = self.switching_env.num_actions
            distr_dim = self.switching_env.num_obs
            obs_dim = n_distributions * distr_dim
            low_dim = self.switching_env.num_states
            n_tasks = total_horizon
            use_true_means = True
            use_svd = True

            # sparse transition matrix
            T = self.switching_env.transition_matrix.T

            # initial state distribution
            pi = np.ones(n_models) / n_models

            # random observation conditional means (assume probabilities of a multinomial distribution)
            O = np.empty((obs_dim, n_models))
            for state in range(n_models):
                curr_state_ac_obs = self.switching_env.state_action_reward_matrix[state, :, :].reshape(-1)
                O[:, state] = curr_state_ac_obs

            """
            # deterministic multinomials
            for i in range(n_distributions):
                for j in range(n_models):
                    v = np.zeros(distr_dim)
                    v[np.random.choice(distr_dim,
                                       p=[0.6, 0.2, 0.1, 0.05, 0.05])] = 1
                    # v[np.random.choice(distr_dim, p=[0.35, 0.2, 0.1, 0.15, 0.1, 0.05, 0.05])] = 1
                    O[i * distr_dim:(i + 1) * distr_dim, j] = v

            # add noise (almost deterministic)
            O += 0.1


            # normalize columns
            for i in range(n_distributions):
                O[i * distr_dim:(i + 1) * distr_dim, :] = O[i * distr_dim:(
                                                                                      i + 1) * distr_dim,
                                                          :] / np.sum(
                    O[i * distr_dim:(i + 1) * distr_dim, :], axis=0)[
                                                               np.newaxis, :]
            """

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
            general_obs_1 = np.zeros((obs_dim, n_tasks))
            general_obs_2 = np.zeros((obs_dim, n_tasks))
            general_obs_3 = np.zeros((obs_dim, n_tasks))

            # Generate tasks/observations
            for j in range(n_tasks):

                # generate next three tasks
                if j == 0:
                    theta_1 = np.random.choice(n_models,
                                               p=pi)  # get random initial task
                else:
                    theta_1 = np.random.choice(n_models, p=T[:,
                                                           theta_3])  # random next task

                theta_2 = np.random.choice(n_models, p=T[:, theta_1])
                theta_3 = np.random.choice(n_models, p=T[:, theta_2])

                # get three observations
                if use_true_means:
                    general_obs_1[:, j] = O[:, theta_1]
                    general_obs_2[:, j] = O[:, theta_2]
                    general_obs_3[:, j] = O[:, theta_3]
                else:
                    obs1 = O[:, theta_1] / O[:, theta_1].sum()
                    o1 = np.random.choice(obs_dim, size=1, p=obs1)[0]
                    general_obs_1[o1, j] = 1

                    obs2 = O[:, theta_2] / O[:, theta_2].sum()
                    o2 = np.random.choice(obs_dim, size=1, p=obs2)[0]
                    general_obs_2[o2, j] = 1

                    obs3 = O[:, theta_3] / O[:, theta_3].sum()
                    o3 = np.random.choice(obs_dim, size=1, p=obs3)[0]
                    general_obs_3[o3, j] = 1

                if j % 5000 == 0:
                    print(j)

            for num_samples in experiments_samples:
                result_dict = {'num_samples': num_samples}

                obs_1 = general_obs_1[:, :num_samples]
                obs_2 = general_obs_2[:, :num_samples]
                obs_3 = general_obs_3[:, :num_samples]

                if use_svd:
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

                T_hat = np.maximum(T_hat, 0)
                T_hat = T_hat / T_hat.sum(axis=0)[None, :]

                O_hat = np.maximum(O_hat, 0)
                for i in range(n_distributions):
                    subset_O_hat = O_hat[i*distr_dim:(i+1)*distr_dim, :]
                    norm_subsest = subset_O_hat / subset_O_hat.sum(axis=0)[None, :]
                    O_hat[i*distr_dim:(i+1)*distr_dim, :] = norm_subsest

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

                result_dict['hmm_o_error'] = max(err_O)
                result_dict['hmm_t_error'] = max(err_T)
                result_dict['min_t_permutation'] = min_t_permutation
                result_dict['min_o_permutation'] = min_o_permutation
                result_dict['norm_1_t_error'] = min_t_error
                result_dict['norm_1_o_error_norm'] = min_o_error / (n_distributions * n_models)
                result_dict['T_hat'] = T_hat.tolist()
                result_dict['O_hat'] = O_hat.tolist()

                result_list.append(result_dict)

        if self.save_bandit_info:
            f = open(self.bandit_dir_path + '/bandit_info.json', 'w')
            json_file = json.dumps(bandit_info_dict)
            f.write(json_file)
            f.close()

        if self.save_results:
            f = open(self.exp_dir_path + f'/exp_info.json', 'w')
            json_file = json.dumps(result_list)
            f.write(json_file)
            f.close()

            """
            for i, elem in enumerate(num_selected_arms_list):
                f = open(self.exp_dir_path + f'/{elem}_arm.json', 'w')
                current_dict = list_of_iteration_dict[i]
                current_dict['result'] = run_result[i].tolist()
                json_file = json.dumps(current_dict)
                f.write(json_file)
                f.close()
            """

            # plt.savefig(self.exp_dir_path + '/plot_regret')

        # plt.show()
