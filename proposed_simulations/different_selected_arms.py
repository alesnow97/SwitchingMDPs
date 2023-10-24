from policies.epsilon_greedy import EpsilonGreedy
from policies.exp3S import Exp3S
from policies.oracle import OraclePolicy
from policies.our_policy import OurPolicy
from policies.sliding_window_UCB import SlidingWindowUCB
from switchingBanditEnv import SwitchingBanditEnv
from matplotlib import pyplot as plt
import numpy as np
import os
import json


class DifferentSelectedArms:

    def __init__(self, switching_env: SwitchingBanditEnv,
                 num_experiments, save_results, loaded_bandit,
                 save_bandit_info=False, bandit_num=0):
        self.switching_env = switching_env
        self.max_reward = self.switching_env.possible_rewards.max()
        self.num_experiments = num_experiments
        self.bandit_dir_path = None
        self.new_bandit_index = None
        self.loaded_bandit = loaded_bandit
        self.bandit_num = bandit_num
        self.generate_dirs()
        self.save_results = save_results
        # if self.loaded_bandit is False:
        #     self.save_bandit_info = True
        # else:
        self.save_bandit_info = save_bandit_info

    def generate_dirs(self):
        dir_name = f"experiments_selected_arms/{self.switching_env.num_states}states_{self.switching_env.num_actions}actions_{self.switching_env.num_obs}obs"
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
        self.exp_dir_path = self.bandit_dir_path + f'/exp{self.new_exp_index}'
        os.mkdir(self.exp_dir_path)

    def run(self, total_horizon, compute_regret_exploitation_horizon=False):
        states_list = np.empty(shape=(self.num_experiments, total_horizon))

        oracle_list = np.empty(shape=(self.num_experiments, total_horizon, 2))
        our_policy0_list = np.empty(shape=(self.num_experiments, total_horizon, 2))
        our_policy1_list = np.empty(shape=(self.num_experiments, total_horizon, 2))
        our_policy2_list = np.empty(shape=(self.num_experiments, total_horizon, 2))
        our_policy3_list = np.empty(shape=(self.num_experiments, total_horizon, 2))

        transition_matrix = self.switching_env.transition_matrix
        state_action_reward_matrix = self.switching_env.state_action_reward_matrix
        possible_rewards = self.switching_env.possible_rewards

        # TODO
        exploration_horizon = 10 * np.sqrt(self.switching_env.num_states) * total_horizon**(2/3)
        print(f"Exploration horizon is {exploration_horizon}")

        bandit_info_dict = {
            'transition_matrix': self.switching_env.transition_matrix.tolist(),
            'reference_matrix': self.switching_env.reference_matrix.tolist(),
            'state_action_reward_matrix': self.switching_env.state_action_reward_matrix.tolist(),
            'possible_rewards': self.switching_env.possible_rewards.tolist(),
            'num_states': self.switching_env.num_states,
            'num_actions': self.switching_env.num_actions,
            'num_obs': self.switching_env.num_obs}

        result_dict = {'total_horizon': total_horizon,
                       'exploration_horizon': exploration_horizon,
                       'num_experiments': self.num_experiments,
                       'rewards': {}}

        our_policy0 = OurPolicy(switching_env=self.switching_env,
                                total_horizon=total_horizon,
                                exploration_horizon=exploration_horizon,
                                num_selected_actions=1)
        our_policy1 = OurPolicy(switching_env=self.switching_env,
                                total_horizon=total_horizon,
                                exploration_horizon=exploration_horizon,
                                num_selected_actions=2)
        our_policy2 = OurPolicy(switching_env=self.switching_env,
                                total_horizon=total_horizon,
                                exploration_horizon=exploration_horizon,
                                num_selected_actions=3)
        our_policy3 = OurPolicy(switching_env=self.switching_env,
                                total_horizon=total_horizon,
                                exploration_horizon=exploration_horizon,
                                num_selected_actions=4)

        for n in range(self.num_experiments):
            print("experiment_n: " + str(n))
            current_state = None

            oracle_policy = OraclePolicy(switching_env=self.switching_env)
            our_policy0.reset()
            our_policy1.reset()
            our_policy2.reset()
            our_policy3.reset()

            for i in range(total_horizon):
                if i == 0:
                    current_state = np.random.randint(low=0, high=self.switching_env.num_states)
                else:
                    current_state = np.random.multinomial(
                            n=1, pvals=transition_matrix[current_state],
                            size=1)[0].argmax()

                # oracle policy with known transition matrix
                oracle_chosen_arm = oracle_policy.choose_arm()
                reward_dist = state_action_reward_matrix[current_state, oracle_chosen_arm]
                reward_index = np.random.multinomial(1, reward_dist, 1)[0].argmax()
                oracle_reward = possible_rewards[reward_index]
                oracle_policy.update(oracle_chosen_arm, reward_index)
                oracle_list[n, i] = [oracle_chosen_arm, oracle_reward]

                # our policy using explore than commit alg
                our_policy_chosen_arm = our_policy0.choose_arm()
                reward_dist = state_action_reward_matrix[current_state, our_policy_chosen_arm]
                reward_index = np.random.multinomial(1, reward_dist, 1)[0].argmax()
                our_policy_reward = possible_rewards[reward_index]
                our_policy0.update(our_policy_chosen_arm, reward_index)
                our_policy0_list[n, i] = [our_policy_chosen_arm, our_policy_reward]

                our_policy_chosen_arm = our_policy1.choose_arm()
                reward_dist = state_action_reward_matrix[current_state, our_policy_chosen_arm]
                reward_index = np.random.multinomial(1, reward_dist, 1)[0].argmax()
                our_policy_reward = possible_rewards[reward_index]
                our_policy1.update(our_policy_chosen_arm, reward_index)
                our_policy1_list[n, i] = [our_policy_chosen_arm, our_policy_reward]

                our_policy_chosen_arm = our_policy2.choose_arm()
                reward_dist = state_action_reward_matrix[current_state, our_policy_chosen_arm]
                reward_index = np.random.multinomial(1, reward_dist, 1)[0].argmax()
                our_policy_reward = possible_rewards[reward_index]
                our_policy2.update(our_policy_chosen_arm, reward_index)
                our_policy2_list[n, i] = [our_policy_chosen_arm, our_policy_reward]

                our_policy_chosen_arm = our_policy3.choose_arm()
                reward_dist = state_action_reward_matrix[current_state, our_policy_chosen_arm]
                reward_index = np.random.multinomial(1, reward_dist, 1)[0].argmax()
                our_policy_reward = possible_rewards[reward_index]
                our_policy3.update(our_policy_chosen_arm, reward_index)
                our_policy3_list[n, i] = [our_policy_chosen_arm, our_policy_reward]

                states_list[n, i] = current_state

                if i % 10000 == 0:
                    print(f"{i}-th epsisode")

        oracle_rewards = oracle_list[:, :, 1]

        result_dict['rewards']['oracle'] = oracle_list.tolist()

        result_dict['rewards']['our_policy0'] = our_policy0_list.tolist()
        result_dict['rewards']['our_policy1'] = our_policy1_list.tolist()
        result_dict['rewards']['our_policy2'] = our_policy2_list.tolist()
        result_dict['rewards']['our_policy3'] = our_policy3_list.tolist()

        our_policy0_regret = np.mean(oracle_rewards - our_policy0_list[:, :, 1], axis=0)
        our_policy1_regret = np.mean(oracle_rewards - our_policy1_list[:, :, 1], axis=0)
        our_policy2_regret = np.mean(oracle_rewards - our_policy2_list[:, :, 1], axis=0)
        our_policy3_regret = np.mean(oracle_rewards - our_policy3_list[:, :, 1], axis=0)

        print("Regret algorithms")
        print(our_policy0_regret.sum())
        print(our_policy1_regret.sum())
        print(our_policy2_regret.sum())
        print(our_policy3_regret.sum())

        plt.plot(np.cumsum(our_policy0_regret), 'r')
        plt.plot(np.cumsum(our_policy1_regret), 'b')
        plt.plot(np.cumsum(our_policy2_regret), 'g')
        plt.plot(np.cumsum(our_policy3_regret), 'c')

        plt.legend([f'{1}_arms', f'{2}_arms', f'{3}_arms', f'{4}_arms'])

        if self.save_bandit_info:
            f = open(self.bandit_dir_path + '/bandit_info.json', 'w')
            json_file = json.dumps(bandit_info_dict)
            f.write(json_file)
            f.close()

        if self.save_results:
            f = open(self.exp_dir_path + '/exp_info.json', 'w')
            json_file = json.dumps(result_dict)
            f.write(json_file)
            f.close()

            plt.savefig(self.exp_dir_path + '/plot_regret')

        plt.show()
