from policies.our_policy_optimistic import OurPolicyOptimistic
from switchingBanditEnv import SwitchingBanditEnv
import numpy as np
import os
import json


class Optimistic:

    def __init__(self, switching_env: SwitchingBanditEnv,
                 num_experiments, save_results, loaded_bandit,
                 save_bandit_info=False, bandit_num=0):
        self.switching_env = switching_env
        self.num_experiments = num_experiments
        self.bandit_dir_path = None
        self.new_bandit_index = None
        self.loaded_bandit = loaded_bandit
        self.bandit_num = bandit_num
        self.generate_dirs()
        self.save_results = save_results
        self.save_bandit_info = save_bandit_info

    def generate_dirs(self):
        dir_name = f"experiments_optimistic/{self.switching_env.num_states}states_{self.switching_env.num_actions}actions_{self.switching_env.num_obs}obs"
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

    def run(self, starting_checkpoint, num_checkpoints, checkpoint_duration):
        transition_matrix = self.switching_env.transition_matrix
        state_action_reward_matrix = self.switching_env.state_action_reward_matrix

        print(f"Starting checkpoint is {starting_checkpoint}")

        bandit_info_dict = {
            'transition_matrix': self.switching_env.transition_matrix.tolist(),
            'reference_matrix': self.switching_env.reference_matrix.tolist(),
            'state_action_reward_matrix': self.switching_env.state_action_reward_matrix.tolist(),
            'num_states': self.switching_env.num_states,
            'num_actions': self.switching_env.num_actions,
            'num_obs': self.switching_env.num_obs}

        result_dict = {'starting_checkpoint': starting_checkpoint,
                       'num_checkpoints': num_checkpoints,
                       'checkpoint_duration': checkpoint_duration,
                       'num_experiments': self.num_experiments}

        # num_selected_arms_list = [1, 2, 3, 5]
        num_selected_arms_list = [3]

        self.num_checkpoints = num_checkpoints
        total_horizon = starting_checkpoint + (self.num_checkpoints - 1) * checkpoint_duration

        run_result = np.empty((2, self.num_experiments, self.num_checkpoints, 2))

        our_policy = OurPolicyOptimistic(switching_env=self.switching_env,
                                         total_horizon=total_horizon,
                                         num_selected_actions=self.switching_env.num_actions)

        result_dict['num_selected_arms'] = num_selected_arms_list[0]
        result_dict['min_selected_arms'] = our_policy.min_exploration_actions
        result_dict['min_complexity_value'] = our_policy.min_complexity_value
        result_dict['min_min_singular_value'] = our_policy.min_min_singular_value
        result_dict['max_exploration_actions'] = our_policy.max_exploration_actions
        result_dict['max_complexity_value'] = our_policy.max_complexity_value
        result_dict['max_min_singular_value'] = our_policy.max_min_singular_value

        for n in range(self.num_experiments):
            print("experiment_n: " + str(n))

            our_policy.reset()
            current_state = None

            for i in range(total_horizon+1):
                if i == 0:
                    current_state = np.random.randint(low=0, high=self.switching_env.num_states)
                else:
                    current_state = np.random.multinomial(
                            n=1, pvals=transition_matrix[current_state],
                            size=1)[0].argmax()

                # our policy using explore than commit alg
                min_chosen_arm, max_chosen_arm = our_policy.choose_arm()
                min_reward_dist = state_action_reward_matrix[current_state, min_chosen_arm]
                max_reward_dist = state_action_reward_matrix[current_state, max_chosen_arm]
                min_reward_index = np.random.multinomial(1, min_reward_dist, 1)[0].argmax()
                max_reward_index = np.random.multinomial(1, max_reward_dist, 1)[0].argmax()
                # our_policy_reward = possible_rewards[reward_index]
                our_policy.update(min_chosen_arm, min_reward_index, max_chosen_arm, max_reward_index)
                # our_policy0_list[n, i] = [our_policy_chosen_arm, our_policy_reward]


                # states_list[n, i] = current_state

                if i % 10000 == 0:
                    print(f"{i}-th epsisode")

            run_result[0, n] = our_policy.min_estimation_errors
            run_result[1, n] = our_policy.max_estimation_errors

        if self.save_bandit_info:
            f = open(self.bandit_dir_path + '/bandit_info.json', 'w')
            json_file = json.dumps(bandit_info_dict)
            f.write(json_file)
            f.close()

        if self.save_results:
            f = open(self.exp_dir_path + f'/exp_info.json', 'w')
            json_file = json.dumps(result_dict)
            f.write(json_file)
            f.close()
            f = open(self.exp_dir_path + f'/min_arm.json', 'w')
            json_file = json.dumps(run_result[0].tolist())
            f.write(json_file)
            f.close()
            f = open(self.exp_dir_path + f'/max_arm.json', 'w')
            json_file = json.dumps(run_result[1].tolist())
            f.write(json_file)
            f.close()
