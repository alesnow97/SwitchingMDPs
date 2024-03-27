import json
import os

import numpy as np

from policies.belief_based_policy import BeliefBasedPolicy
from pomdp_env import POMDP
from strategy.optimisticAlgorithmStrategy import OptimisticAlgorithmStrategy
from strategy.oracleStrategy import OracleStrategy
from strategy.rebuttalEstErrExp import EstimationErrorStrategy

class POMDPSimulationNew:

    def __init__(self,
                 pomdp: POMDP,
                 loaded_pomdp,
                 pomdp_num=0,
                 save_pomdp_info=True,
                 save_basic_info=True,
                 save_results=True
                 ):

        self.pomdp = pomdp
        self.loaded_pomdp = loaded_pomdp
        self.pomdp_num = pomdp_num
        self.num_states = self.pomdp.num_states
        self.num_actions = self.pomdp.num_actions
        self.num_obs = self.pomdp.num_obs

        self.save_pomdp_info = save_pomdp_info
        self.save_basic_info = save_basic_info
        self.save_results = save_results


    def generate_dirs(self, experiment_type):

        dir_name = f"ICML_experiments/{self.num_states}states_{self.num_actions}actions_{self.num_obs}obs"

        if os.path.exists(dir_name):
            if self.loaded_pomdp:
                self.pomdp_dir_path = dir_name + f'/pomdp{self.pomdp_num}'
                self.exp_type_path = self.pomdp_dir_path + f"/{experiment_type}"
                self.new_exp_index = len(os.listdir(self.exp_type_path))
                print(self.new_exp_index)
            else:
                self.new_pomdp_index = len(os.listdir(dir_name))
                self.pomdp_dir_path = dir_name + f'/pomdp{self.new_pomdp_index}'
                os.mkdir(self.pomdp_dir_path)
                est_error_exp_path = self.pomdp_dir_path + "/estimation_error"
                regret_exp_path = self.pomdp_dir_path + "/regret"
                os.mkdir(est_error_exp_path)
                os.mkdir(regret_exp_path)
                self.exp_type_path = self.pomdp_dir_path + f"/{experiment_type}"
                self.new_exp_index = 0
        else:
            os.mkdir(dir_name)
            self.new_pomdp_index = 0
            self.pomdp_dir_path = dir_name + f'/pomdp{self.new_pomdp_index}'
            os.mkdir(self.pomdp_dir_path)
            est_error_exp_path = self.pomdp_dir_path + "/estimation_error"
            regret_exp_path = self.pomdp_dir_path + "/regret"
            os.mkdir(est_error_exp_path)
            os.mkdir(regret_exp_path)
            self.exp_type_path = self.pomdp_dir_path + f"/{experiment_type}"
            self.new_exp_index = 0
        # self.exp_dir_path = self.exp_type_path + f'/exp{self.new_exp_index}'
        # os.mkdir(self.exp_dir_path)


    def run_estimation_error(self, num_experiments: int, num_samples_to_discard: int,
            num_samples_checkpoint: int, num_checkpoints: int):

        # the different types of experiment types are "estimation_error" and "regret"
        self.generate_dirs(experiment_type="estimation_error")

        pomdp_info_dict = self.pomdp.generate_pomdp_dict()

        result_dict = {'num_samples_to_discard': num_samples_to_discard,
                       'num_checkpoints': num_checkpoints,
                       'num_samples_checkpoint': num_samples_checkpoint,
                       'num_experiments': num_experiments}

        # mode_state_action_rew_list = np.empty(
        #     shape=(num_experiments, horizon, 4), dtype=int)


        # state_action_reward_matrix = self.switching_env.state_action_reward_matrix
        # possible_rewards = self.switching_env.possible_rewards

        self.estimated_action_state_dist = np.zeros(shape=(num_experiments,
            num_checkpoints, self.num_actions, self.num_actions,
            self.num_states, self.num_states))

        self.estimated_transition_matrix = np.zeros(shape=(num_experiments,
            num_checkpoints, self.num_states, self.num_actions, self.num_states))

        self.error_frobenious_norm = np.empty(shape=(num_experiments, num_checkpoints))

        for n in range(num_experiments):
            print("Experiment_n: " + str(n))

            initial_state = np.random.random_integers(low=0, high=self.num_states-1)

            # used policy
            self.policy = BeliefBasedPolicy(
                self.num_states, self.num_actions, self.num_obs,
                self.pomdp.state_action_transition_matrix,
                self.pomdp.state_action_observation_matrix,
                self.pomdp.possible_rewards
            )

            self.estimation_error_strategy = EstimationErrorStrategy(
                num_states=self.num_states,
                num_actions=self.num_actions,
                num_obs=self.num_obs,
                pomdp=self.pomdp,
                policy = self.policy,
            )

            (estimated_action_state_dist,
             estimated_transition_matrix,
             error_frobenious_norm) = self.estimation_error_strategy.run(
                num_samples_to_discard=num_samples_to_discard,
                num_samples_checkpoint=num_samples_checkpoint,
                num_checkpoints=num_checkpoints,
                initial_state=initial_state)

            self.estimated_action_state_dist[n] = estimated_action_state_dist
            self.estimated_transition_matrix[n] = estimated_transition_matrix
            self.error_frobenious_norm[n] = error_frobenious_norm

        result_dict['estimated_action_state_dist'] = self.estimated_action_state_dist.tolist()
        result_dict['estimated_transition_matrix'] = self.estimated_transition_matrix.tolist()
        result_dict['error_frobenious_norm'] = self.error_frobenious_norm.tolist()

        if not self.loaded_pomdp and self.save_pomdp_info:
            f = open(self.pomdp_dir_path + '/pomdp_info.json', 'w')
            json_file = json.dumps(pomdp_info_dict)
            f.write(json_file)
            f.close()

        if self.save_results:
            f = open(self.exp_type_path + f'/{num_samples_checkpoint}_{num_checkpoints}cp_{self.new_exp_index}.json', 'w')
            json_file = json.dumps(result_dict)
            f.write(json_file)
            f.close()
            # for i, elem in enumerate(num_selected_arms_list):
            #     f = open(self.exp_dir_path + f'/{elem}_arm.json', 'w')
            #     current_dict = list_of_iteration_dict[i]
            #     current_dict['result'] = run_result[i].tolist()
            #     json_file = json.dumps(current_dict)
            #     f.write(json_file)
            #     f.close()


    def run_regret_experiment(self,
                              num_experiments: int,
                              T_0: int,
                              num_episodes: int,
                              ext_v_i_stopping_cond: float,
                              state_discretization_step: float,
                              action_discretization_step: float,
                              min_action_prob: float,
                              delta: float,
                              run_oracle: bool,
                              run_optimistic: bool,
                              discretized_belief_states: np.ndarray = None,
                              discretized_action_space: np.ndarray = None,
                              real_belief_action_belief: np.ndarray = None,
                              real_optimal_belief_action_mapping: np.ndarray = None,
                              initial_discretized_belief: np.ndarray = None,
                              initial_discretized_belief_index: int = None,
                              ):

        # disable this for the moment
        self.generate_dirs(experiment_type="regret")

        pomdp_info_dict = self.pomdp.generate_pomdp_dict()

        # self.estimated_action_state_dist = np.zeros(shape=(num_experiments,
        #     num_checkpoints, self.num_actions, self.num_actions,
        #     self.num_states, self.num_states))
        #
        # self.estimated_transition_matrix = np.zeros(shape=(num_experiments,
        #     num_checkpoints, self.num_states, self.num_actions, self.num_states))
        #
        # self.error_frobenious_norm = np.empty(shape=(num_experiments, num_checkpoints))

        self.oracle_strategy = OracleStrategy(
            num_states=self.num_states,
            num_actions=self.num_actions,
            num_obs=self.num_obs,
            pomdp=self.pomdp,
            ext_v_i_stopping_cond=ext_v_i_stopping_cond,
            epsilon_state=state_discretization_step,
            epsilon_action=action_discretization_step,
            min_action_prob=min_action_prob,
            discretized_belief_states=discretized_belief_states,
            discretized_action_space=discretized_action_space,
            real_belief_action_belief=real_belief_action_belief,
            real_optimal_belief_action_mapping=real_optimal_belief_action_mapping,
            initial_discretized_belief=initial_discretized_belief,
            initial_discretized_belief_index=initial_discretized_belief_index,
        )

        self.optimistic_algorithm_strategy = OptimisticAlgorithmStrategy(
            num_states=self.num_states,
            num_actions=self.num_actions,
            num_obs=self.num_obs,
            pomdp=self.pomdp,
            ext_v_i_stopping_cond=ext_v_i_stopping_cond,
            epsilon_state=state_discretization_step,
            epsilon_action=action_discretization_step,
            min_action_prob=min_action_prob,
            delta=delta
        )

        oracle_strategy_basic_info_dict = self.oracle_strategy.generate_basic_info_dict()
        # optimistic_strategy_basic_info_dict = self.optimistic_algorithm_strategy.generate_basic_info_dict()

        oracle_collected_samples = None
        optimistic_alg_collected_samples = None
        estimated_transition_matrices = np.zeros(shape=(num_experiments,
            num_episodes, self.num_states, self.num_actions,
            self.num_states))
        frobenious_norm_error = np.zeros(shape=(num_experiments, num_episodes))

        for n in range(num_experiments):
            print("Experiment_n: " + str(n))


            initial_state = np.random.multinomial(1, np.ones(shape=self.num_states) / self.num_states, 1)[
                0].argmax()

            if run_oracle is True:
                oracle_collected_samples_per_exp = self.oracle_strategy.run(
                    T_0=T_0,
                    num_episodes=num_episodes,
                    initial_state=initial_state,
                )

                if oracle_collected_samples is None:
                    num_collected_samples = \
                    oracle_collected_samples_per_exp.shape[0]
                    oracle_collected_samples = np.zeros(
                        shape=(num_experiments, num_collected_samples, 3))
                oracle_collected_samples[n] = oracle_collected_samples_per_exp

            if run_optimistic is True:
                opt_collected_samples_per_exp, estimated_trans_mat_per_exp, frobenious_norm_per_exp = (
                    self.optimistic_algorithm_strategy.run(
                    T_0=T_0,
                    num_episodes=num_episodes,
                    initial_state=initial_state,
                ))

                if optimistic_alg_collected_samples is None:
                    num_collected_samples = opt_collected_samples_per_exp.shape[0]
                    optimistic_alg_collected_samples = np.zeros(shape=(num_experiments, num_collected_samples, 3))

                optimistic_alg_collected_samples[
                    n] = opt_collected_samples_per_exp

                # oracle_collected_samples.append(oracle_collected_samples_per_exp)
                # optimistic_alg_collected_samples.append(opt_collected_samples_per_exp)
                estimated_transition_matrices[n] = estimated_trans_mat_per_exp
                frobenious_norm_error[n] = frobenious_norm_per_exp

        oracle_result_dict = None
        optimistic_result_dict = None
        if run_oracle:
            oracle_result_dict = {
                'T_0': T_0,
                'num_episodes': num_episodes,
                'num_experiments': num_experiments,
                "oracle_collected_samples": oracle_collected_samples.tolist(),
            }

        if run_optimistic:
            optimistic_result_dict = {
                'T_0': T_0,
                'num_episodes': num_episodes,
                'num_experiments': num_experiments,
                "delta": delta,
                "optimistic_alg_collected_samples": optimistic_alg_collected_samples.tolist(),
                "estimated_transition_matrices": estimated_transition_matrices.tolist(),
                "frobenious_norm_error": frobenious_norm_error.tolist()
            }

        if not self.loaded_pomdp and self.save_pomdp_info:
            f = open(self.pomdp_dir_path + '/pomdp_info.json', 'w')
            json_file = json.dumps(pomdp_info_dict)
            f.write(json_file)
            f.close()

        if discretized_belief_states is None and self.save_basic_info:
            basic_info_path = f"/{state_discretization_step}stst_{action_discretization_step}acst_{min_action_prob}_minac"
            dir_to_create_path = self.exp_type_path + basic_info_path
            if not os.path.exists(dir_to_create_path):
                os.mkdir(dir_to_create_path)
            f = open(
                dir_to_create_path + f'/basic_info.json',
                'w')
            json_file = json.dumps(oracle_strategy_basic_info_dict)
            f.write(json_file)
            f.close()
            print("Basic info have been saved")

        if self.save_results:
            basic_info_path = f"/{state_discretization_step}stst_{action_discretization_step}acst_{min_action_prob}_minac"
            dir_to_create_path = self.exp_type_path + basic_info_path
            if os.path.exists(dir_to_create_path):
                new_exp_index = len(os.listdir(dir_to_create_path))
                if run_oracle:
                    f = open(
                        dir_to_create_path + f'/oracle_{np.log(T_0)}Init_{num_episodes}Ep_{num_experiments}Exp_{new_exp_index-1}.json',
                        'w')
                    json_file = json.dumps(oracle_result_dict)
                    f.write(json_file)
                    f.close()
                    print("Oracle Results have been saved")
                if run_optimistic:
                    f = open(
                        dir_to_create_path + f'/optimistic_{np.log(T_0)}Init_{num_episodes}Ep_{num_experiments}Exp_{new_exp_index-1}.json',
                        'w')
                    json_file = json.dumps(optimistic_result_dict)
                    f.write(json_file)
                    f.close()
                    print("Optimistic Algorithm Results have been saved")
            else:
                raise ValueError("The folder does not exist")
