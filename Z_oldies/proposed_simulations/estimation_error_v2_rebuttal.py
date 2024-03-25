from Z_oldies.policies.our_policy_estimation_error_v2_rebuttal import \
    OurPolicyTweakedV2Rebuttal
from Z_oldies.switchingBanditEnv import SwitchingBanditEnv
import numpy as np
import os
import json


class EstimationErrorExpV2Rebuttal:

    def __init__(self, switching_env: SwitchingBanditEnv,
                 num_experiments, save_results, loaded_bandit, dir_name,
                 save_bandit_info=False, bandit_num=0):
        self.switching_env = switching_env
        self.num_experiments = num_experiments
        self.bandit_dir_path = None
        self.new_bandit_index = None
        self.dir_name = dir_name
        self.loaded_bandit = loaded_bandit
        self.bandit_num = bandit_num
        self.generate_dirs()
        self.save_results = save_results
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
        self.exp_dir_path = self.bandit_dir_path + f'/exp{self.new_exp_index}'
        os.mkdir(self.exp_dir_path)

    def run(self, experiments_samples, num_arms):
        transition_matrix = self.switching_env.transition_matrix
        state_action_reward_matrix = self.switching_env.state_action_reward_matrix

        bandit_info_dict = {
            'transition_matrix': self.switching_env.transition_matrix.tolist(),
            'reference_matrix': self.switching_env.reference_matrix.tolist(),
            'state_action_reward_matrix': self.switching_env.state_action_reward_matrix.tolist(),
            'num_states': self.switching_env.num_states,
            'num_actions': self.switching_env.num_actions,
            'num_obs': self.switching_env.num_obs}

        num_selected_arms_list = [num_arms]
        self.num_checkpoints = len(experiments_samples)
        self.total_horizon = experiments_samples[-1]

        our_policy = OurPolicyTweakedV2Rebuttal(switching_env=self.switching_env,
                                      experiments_samples=experiments_samples,
                                      num_selected_actions=num_selected_arms_list[0])

        result_dict = {}
        result_dict['num_selected_arms'] = num_selected_arms_list[0]
        result_dict['min_selected_arms'] = our_policy.min_exploration_actions
        result_dict['min_complexity_value'] = our_policy.min_complexity_value
        result_dict['min_min_singular_value'] = our_policy.min_min_singular_value
        result_dict['experiments_samples'] = experiments_samples

        current_state = None
        for i in range(self.total_horizon+1):
            if i == 0:
                current_state = np.random.randint(low=0, high=self.switching_env.num_states)
            else:
                current_state = np.random.multinomial(
                        n=1, pvals=transition_matrix[current_state],
                        size=1)[0].argmax()

            # our policy using explore than commit alg
            min_chosen_arm = our_policy.choose_arm()
            min_reward_dist = state_action_reward_matrix[current_state, min_chosen_arm]
            min_reward_index = np.random.multinomial(1, min_reward_dist, 1)[0].argmax()
            # our_policy_reward = possible_rewards[reward_index]
            our_policy.update(min_chosen_arm, min_reward_index)
            # our_policy0_list[n, i] = [our_policy_chosen_arm, our_policy_reward]


            # states_list[n, i] = current_state

            if i % 10000 == 0:
                print(f"{i}-th epsisode")

        result_dict["estimation_errors"] = our_policy.min_estimation_errors.tolist()

        if self.save_bandit_info:
            f = open(self.bandit_dir_path + '/bandit_info.json', 'w')
            json_file = json.dumps(bandit_info_dict)
            f.write(json_file)
            f.close()

        if self.save_results:
            f = open(self.exp_dir_path + f'/exp_our_info.json', 'w')
            json_file = json.dumps(result_dict)
            f.write(json_file)
            f.close()
