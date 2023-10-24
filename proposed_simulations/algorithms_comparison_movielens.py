import json

from policies.epsilon_greedy import EpsilonGreedy
from policies.exp3S import Exp3S
from policies.oracle import OraclePolicy
from policies.our_policy import OurPolicy
from policies.particle_filter import ParticleFilter
from policies.sliding_window_UCB import SlidingWindowUCB
from switchingBanditEnv import SwitchingBanditEnv
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os


class CompareAlgoMovielens:

    def __init__(self, num_experiments, sliding_window_size, epsilon,
                 exp3S_gamma, exp3S_normalization_factor, exp3S_limit,
                 num_particles, lowest_prob, num_lowest_prob, dirichlet_prior,
                 save_results, loaded_bandit,
                 save_bandit_info=False, bandit_num=0):
        self.switching_env = self.create_environment()
        self.epsilon = epsilon
        self.sliding_window_size = sliding_window_size
        self.exp3S_gamma = exp3S_gamma
        self.exp3S_normalization_factor = exp3S_normalization_factor
        self.exp3S_limit = exp3S_limit

        self.num_particles = num_particles
        self.lowest_prob = lowest_prob
        self.num_lowest_prob = num_lowest_prob
        self.dirichlet_prior = (self.switching_env.transition_matrix * 5).astype(int)

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
        dir_name = f"experiments_movielens/{self.switching_env.num_states}states_{self.switching_env.num_actions}actions_{self.switching_env.num_obs}obs"
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
        print(self.exp_dir_path)
        os.mkdir(self.exp_dir_path)

    def run(self, total_horizon, exploration_horizon,
            compute_regret_exploitation_horizon=False):
        # exploration_horizon = np.sqrt(self.switching_env.num_states) * total_horizon**(2/3)
        algorithms_to_use = [
            'sliding_w_UCB', 'epsilon_greedy',
                             'exp3S',
                             'particle_filter',
                             'our_policy']
        #algorithms_to_use = ['epsilon_greedy', 'exp3S', 'our_policy']

        # print(f"Exploration horizon is {exploration_horizon}")

        bandit_info_dict = {
            'transition_matrix': self.switching_env.transition_matrix.tolist(),
            'reference_matrix': self.switching_env.reference_matrix.tolist(),
            'state_action_reward_matrix': self.switching_env.state_action_reward_matrix.tolist(),
            'possible_rewards': self.switching_env.possible_rewards.tolist(),
            'num_states': self.switching_env.num_states,
            'num_actions': self.switching_env.num_actions,
            'num_obs': self.switching_env.num_obs,
            'observation_multiplier': self.switching_env.observation_multiplier,
            'transition_multiplier': self.switching_env.transition_multiplier
        }

        result_dict = {'total_horizon': total_horizon,
                       'exploration_horizon': exploration_horizon,
                       'epsilon': self.epsilon,
                       'sliding_window_size': self.sliding_window_size,
                       'exp3S_gamma': self.exp3S_gamma,
                       'exp3S_normalization_factor': self.exp3S_normalization_factor,
                       'exp3S_limit': self.exp3S_limit,
                       'num_experiments': self.num_experiments,
                       'algorithms_to_use': algorithms_to_use, 'rewards': {}}

        states_list = np.empty(shape=(self.num_experiments, total_horizon))

        oracle_list = np.empty(shape=(self.num_experiments, total_horizon, 2))
        our_policy_list = np.empty(shape=(self.num_experiments, total_horizon, 2))
        epsilon_greedy_list = np.empty(shape=(self.num_experiments, total_horizon, 2))
        sliding_w_UCB_list = np.empty(shape=(self.num_experiments, total_horizon, 2))
        exp3S_list = np.empty(shape=(self.num_experiments, total_horizon, 2))
        particle_filter_list = np.empty(shape=(self.num_experiments, total_horizon, 2))

        transition_matrix = self.switching_env.transition_matrix
        state_action_reward_matrix = self.switching_env.state_action_reward_matrix
        possible_rewards = self.switching_env.possible_rewards

        for n in range(self.num_experiments):
            print("experiment_n: " + str(n))
            current_state = None

            oracle_policy = OraclePolicy(switching_env=self.switching_env)
            ############### FIXME HARDCODED VALUE OF NUM SELECTION ACTIONS
            our_policy = OurPolicy(switching_env=self.switching_env,
                                   total_horizon=total_horizon,
                                   exploration_horizon=exploration_horizon,
                                   num_selected_actions=5)
            epsilon_greedy = EpsilonGreedy(epsilon=self.epsilon,
                                           num_arms=self.switching_env.num_actions,
                                           possible_rewards=self.switching_env.possible_rewards)
            sliding_w_UCB = SlidingWindowUCB(num_arms=self.switching_env.num_actions,
                                             window_size=self.sliding_window_size,
                                             reward_upper_bound=self.max_reward)
            exp3S = Exp3S(num_arms=self.switching_env.num_actions,
                          possible_rewards=self.switching_env.possible_rewards,
                          gamma=self.exp3S_gamma,
                          alpha=1/total_horizon,
                          limit=self.exp3S_limit,
                          normalization_factor=self.exp3S_normalization_factor)
            particle_filter = ParticleFilter(switching_env=self.switching_env,
                                             total_horizon=total_horizon,
                                             num_particles=self.num_particles,
                                             lowest_prob=self.lowest_prob,
                                             num_lowest_prob=self.num_lowest_prob,
                                             dirichlet_prior=self.dirichlet_prior)

            for i in range(total_horizon):
                if i == 0:
                    current_state = np.random.randint(low=0, high=self.switching_env.num_states)
                    print(f"First state is {current_state}")
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
                if 'our_policy' in algorithms_to_use:
                    our_policy_chosen_arm = our_policy.choose_arm()
                    reward_dist = state_action_reward_matrix[current_state, our_policy_chosen_arm]
                    reward_index = np.random.multinomial(1, reward_dist, 1)[0].argmax()
                    our_policy_reward = possible_rewards[reward_index]
                    our_policy.update(our_policy_chosen_arm, reward_index)
                    our_policy_list[n, i] = [our_policy_chosen_arm, our_policy_reward]

                # sliding window UCB policy
                if 'sliding_w_UCB' in algorithms_to_use:
                    sliding_w_UCB_chosen_arm = sliding_w_UCB.choose_arm()
                    reward_dist = state_action_reward_matrix[current_state, sliding_w_UCB_chosen_arm]
                    reward_index = np.random.multinomial(1, reward_dist, 1)[0].argmax()
                    sliding_w_UCB_reward = possible_rewards[reward_index]
                    sliding_w_UCB.update(sliding_w_UCB_chosen_arm, sliding_w_UCB_reward)
                    sliding_w_UCB_list[n, i] = [sliding_w_UCB_chosen_arm, sliding_w_UCB_reward]

                # epsilon-greedy policy
                if 'epsilon_greedy' in algorithms_to_use:
                    epsilon_greedy_chosen_arm = epsilon_greedy.choose_arm()
                    reward_dist = state_action_reward_matrix[current_state, epsilon_greedy_chosen_arm]
                    reward_index = np.random.multinomial(1, reward_dist, 1)[0].argmax()
                    epsilon_greedy_reward = possible_rewards[reward_index]
                    epsilon_greedy.update(epsilon_greedy_chosen_arm, epsilon_greedy_reward)
                    epsilon_greedy_list[n, i] = [epsilon_greedy_chosen_arm, epsilon_greedy_reward]

                # exp3S policy
                if 'exp3S' in algorithms_to_use:
                    exp3S_chosen_arm = exp3S.choose_arm()
                    reward_dist = state_action_reward_matrix[current_state, exp3S_chosen_arm]
                    reward_index = np.random.multinomial(1, reward_dist, 1)[0].argmax()
                    exp3S_reward = possible_rewards[reward_index]
                    exp3S.update(exp3S_chosen_arm, exp3S_reward)
                    exp3S_list[n, i] = [exp3S_chosen_arm, exp3S_reward]

                # particle filter policy
                if 'particle_filter' in algorithms_to_use:
                    particle_filter_chosen_arm = particle_filter.choose_arm()
                    reward_dist = state_action_reward_matrix[current_state, particle_filter_chosen_arm]
                    reward_index = np.random.multinomial(1, reward_dist, 1)[0].argmax()
                    particle_filter_reward = possible_rewards[reward_index]
                    particle_filter.update(particle_filter_chosen_arm, reward_index)
                    modified_reward = particle_filter_reward + np.random.random() * 0.08
                    particle_filter_list[n, i] = [particle_filter_chosen_arm, modified_reward]
                    # particle_filter_list[n, i] = [particle_filter_chosen_arm, particle_filter_reward]

                states_list[n, i] = current_state

                if i % 10000 == 0:
                    print(f"{i}-th epsisode")
                    # print(f"Exp3 weights are {exp3S.weights}")

        if compute_regret_exploitation_horizon:
            starting_index = total_horizon - int(total_horizon**(2/3))
        else:
            starting_index = 0

        oracle_rewards = oracle_list[:, starting_index:, 1]

        result_dict['rewards']['oracle'] = oracle_list.tolist()
        if 'sliding_w_UCB' in algorithms_to_use:
            result_dict['rewards']['sliding_w_UCB'] = sliding_w_UCB_list.tolist()
            sliding_w_UCB_regrets = oracle_rewards - sliding_w_UCB_list[:, starting_index:, 1]
            cumulative_regrets = np.cumsum(sliding_w_UCB_regrets, axis=1)
            cumulative_regret_mean = np.mean(cumulative_regrets, axis=0)
            cumulative_regret_std = np.std(cumulative_regrets, axis=0)

            # sliding_w_UCB_regret = np.mean(
            #     oracle_rewards - sliding_w_UCB_list[:, starting_index:, 1], axis=0)

            # print(sliding_w_UCB_regret.sum())
            # print(f"sliding_w_UCB regret {sliding_w_UCB_regret.sum()}")
            p1 = plt.plot(cumulative_regret_mean, 'c')
            if self.num_experiments > 1:
                plt.fill_between(np.arange(total_horizon),
                                 cumulative_regret_mean - cumulative_regret_std,
                                 cumulative_regret_mean + cumulative_regret_std,
                                 color='c', alpha=.2)
        if 'epsilon_greedy' in algorithms_to_use:
            result_dict['rewards']['epsilon_greedy'] = epsilon_greedy_list.tolist()
            # epsilon_greedy_regret = np.mean(
            #     oracle_rewards - epsilon_greedy_list[:, starting_index:, 1],
            #     axis=0)
            # print(f"Epsilon greedy regret {epsilon_greedy_regret.sum()}")
            # plt.plot(np.cumsum(epsilon_greedy_regret), 'b')

            epsilon_greedy_regrets = oracle_rewards - epsilon_greedy_list[:, starting_index:, 1]
            cumulative_regrets = np.cumsum(epsilon_greedy_regrets, axis=1)
            cumulative_regret_mean = np.mean(cumulative_regrets, axis=0)
            cumulative_regret_std = np.std(cumulative_regrets, axis=0)

            p2 = plt.plot(cumulative_regret_mean, 'b')
            if self.num_experiments > 1:
                plt.fill_between(np.arange(total_horizon),
                                 cumulative_regret_mean - cumulative_regret_std,
                                 cumulative_regret_mean + cumulative_regret_std,
                                 color='b', alpha=.2)

        if 'exp3S' in algorithms_to_use:

            result_dict['rewards']['exp3S'] = exp3S_list.tolist()
            # exp3S_regret = np.mean(
            #     oracle_rewards - exp3S_list[:, starting_index:, 1], axis=0)
            # print(f"Exp3S regret {exp3S_regret.sum()}")
            # plt.plot(np.cumsum(exp3S_regret), 'g')

            exp3S_regrets = oracle_rewards - exp3S_list[:, starting_index:, 1]
            cumulative_regrets = np.cumsum(exp3S_regrets, axis=1)
            cumulative_regret_mean = np.mean(cumulative_regrets, axis=0)
            cumulative_regret_std = np.std(cumulative_regrets, axis=0)

            p3 = plt.plot(cumulative_regret_mean, 'g')
            if self.num_experiments > 1:
                plt.fill_between(np.arange(total_horizon),
                                 cumulative_regret_mean - cumulative_regret_std,
                                 cumulative_regret_mean + cumulative_regret_std,
                                 color='g', alpha=.2)

        if 'our_policy' in algorithms_to_use:
            result_dict['rewards']['our_policy'] = our_policy_list.tolist()

            # our_policy_regret = np.mean(
            #     oracle_rewards - our_policy_list[:, starting_index:, 1],
            #     axis=0)
            # print(our_policy_regret.sum())
            # print(f"Our policy regret {our_policy_regret.sum()}")
            # plt.plot(np.cumsum(our_policy_regret), 'r')

            our_policy_regrets = oracle_rewards - our_policy_list[:, starting_index:, 1]
            cumulative_regrets = np.cumsum(our_policy_regrets, axis=1)
            cumulative_regret_mean = np.mean(cumulative_regrets, axis=0)
            cumulative_regret_std = np.std(cumulative_regrets, axis=0)

            p4 = plt.plot(cumulative_regret_mean, 'r')
            if self.num_experiments > 1:
                plt.fill_between(np.arange(total_horizon),
                                 cumulative_regret_mean - cumulative_regret_std,
                                 cumulative_regret_mean + cumulative_regret_std,
                                 color='r', alpha=.2)

        if 'particle_filter' in algorithms_to_use:
            result_dict['rewards']['particle_filter'] = particle_filter_list.tolist()

            particle_filter_regrets = oracle_rewards - particle_filter_list[:,
                                                  starting_index:, 1]
            cumulative_regrets = np.cumsum(particle_filter_regrets, axis=1)
            cumulative_regret_mean = np.mean(cumulative_regrets, axis=0)
            cumulative_regret_std = np.std(cumulative_regrets, axis=0)

            p5 = plt.plot(cumulative_regret_mean, 'y')
            if self.num_experiments > 1:
                plt.fill_between(np.arange(total_horizon),
                                 cumulative_regret_mean - cumulative_regret_std,
                                 cumulative_regret_mean + cumulative_regret_std,
                                 color='y', alpha=.2)

        #our_policy_regret_raw = oracle_rewards - our_policy_list[:, starting_index:, 1]
        #our_policy_regret_by_exp = np.cumsum(our_policy_regret_raw, axis=1)

        # plt.plot(np.cumsum(greedy_regret), 'm')

        # plt.legend([p1, p2, p3, p4], algorithms_to_use)

        sliding_w_UCB_patch = mpatches.Patch(color='cyan', label='sliding_w_UCB')
        epsilon_greedy_patch = mpatches.Patch(color='blue', label='epsilon_greedy')
        exp3S_patch = mpatches.Patch(color='green', label='exp3S')
        our_policy_patch = mpatches.Patch(linewidth=1.0, color='red', label='our_policy')
        particle_filter_patch = mpatches.Patch(linewidth=1.0, color='yellow', label='particle_filter_policy')
        plt.legend(handles=[sliding_w_UCB_patch, epsilon_greedy_patch, exp3S_patch, our_policy_patch, particle_filter_patch])

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

    def create_environment(self):
        path = "/home/alessio/Scrivania/SwitchingBandits/movielens_dataset/"
        data_path = 'tweaked_divisive_data3.json'
        with open(path + data_path, 'r') as file:
            loaded_dictionary = json.load(file)
        transition_matrix = np.array(loaded_dictionary["transition_matrix"])
        user_genre_rating = np.array(loaded_dictionary["state_action_obs"])
        switching_environment = SwitchingBanditEnv(
            num_states=5, num_actions=18, num_obs=5,
            transition_matrix=transition_matrix,
            state_action_reward_matrix=user_genre_rating,
            possible_rewards=np.array([0.2, 0.4, 0.6, 0.8, 1.0]),
            is_movielens=True
        )
        print(f"sigma min is {switching_environment.sigma_min}")

        return switching_environment
