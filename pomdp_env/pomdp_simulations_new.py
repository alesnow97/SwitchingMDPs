import json
import os

import numpy as np

from policies.belief_based_policy import BeliefBasedPolicy
from pomdp_env import POMDP
from strategy.PSRL.psrlStrategy import PSRLStrategy
from strategy.optimisticAlgorithmStrategy import OptimisticAlgorithmStrategy
from strategy.oracleStrategy import OracleStrategy
from strategy.rebuttalEstErrExp import EstimationErrorStrategy
from strategy.second_rebuttalEstErrExp import SecondEstimationErrorStrategy
from strategy.spectralAlgorithmStrategy import SpectralAlgorithmStrategy


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

        if experiment_type == "regret":
            base_base = "ICML_experiments"
        else:
            base_base = "ICML_experiments_error"

        dir_name = f"{base_base}/{self.num_states}states_{self.num_actions}actions_{self.num_obs}obs"

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
            num_samples_checkpoint: int, num_checkpoints: int, min_action_prob: float):

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

        # used policy
        self.policy = BeliefBasedPolicy(
            num_states=self.num_states,
            num_actions=self.num_actions,
            num_obs=self.num_obs,
            min_action_prob=min_action_prob,
            state_action_transition_matrix=self.pomdp.state_action_transition_matrix,
            state_action_observation_matrix=self.pomdp.state_action_observation_matrix,
            possible_rewards=self.pomdp.possible_rewards
        )

        for n in range(num_experiments):
            print("Experiment_n: " + str(n))

            initial_state = np.random.random_integers(low=0, high=self.num_states-1)

            self.policy.reset_belief()

            self.estimation_error_strategy = SecondEstimationErrorStrategy(
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

        # result_dict['estimated_action_state_dist'] = self.estimated_action_state_dist.tolist()
        # result_dict['estimated_transition_matrix'] = self.estimated_transition_matrix.tolist()
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
                              min_action_prob: float,
                              delta: float,
                              run_oracle: bool,
                              run_optimistic: bool,
                              starting_episode_num: int = 0,
                              discretized_belief_states: np.ndarray = None,
                              real_belief_action_belief: np.ndarray = None,
                              real_optimal_belief_action_mapping: np.ndarray = None,
                              initial_discretized_belief: np.ndarray = None,
                              initial_discretized_belief_index: int = None,
                              tau_1: int = None,
                              tau_2: int = None,
                              ):

        # just for testing
        run_oracle = False
        run_optimistic = False
        run_spectral = False
        run_psrl = True

        self.generate_dirs(experiment_type="regret")
        self.state_discretization_step = state_discretization_step
        self.min_action_prob = min_action_prob

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
            min_action_prob=min_action_prob,
            discretized_belief_states=discretized_belief_states,
            real_belief_action_belief=real_belief_action_belief,
            real_optimal_belief_action_mapping=real_optimal_belief_action_mapping,
            initial_discretized_belief=initial_discretized_belief,
            initial_discretized_belief_index=initial_discretized_belief_index,
            save_path=self.exp_type_path
        )

        self.optimistic_algorithm_strategy = OptimisticAlgorithmStrategy(
            num_states=self.num_states,
            num_actions=self.num_actions,
            num_obs=self.num_obs,
            pomdp=self.pomdp,
            ext_v_i_stopping_cond=ext_v_i_stopping_cond,
            epsilon_state=state_discretization_step,
            min_action_prob=min_action_prob,
            delta=delta,
            discretized_belief_states=self.oracle_strategy.discretized_belief_states,
            save_path=self.exp_type_path
        )

        self.spectral_algorithm_strategy = SpectralAlgorithmStrategy(
            num_states=self.num_states,
            num_actions=self.num_actions,
            num_obs=self.num_obs,
            pomdp=self.pomdp,
            ext_v_i_stopping_cond=ext_v_i_stopping_cond,
            epsilon_state=state_discretization_step,
            min_action_prob=min_action_prob,
            delta=delta,
            discretized_belief_states=self.oracle_strategy.discretized_belief_states,
            save_path=self.exp_type_path
        )

        self.psrl_algorithm_strategy = PSRLStrategy(
            num_states=self.num_states,
            num_actions=self.num_actions,
            num_obs=self.num_obs,
            pomdp=self.pomdp,
            ext_v_i_stopping_cond=ext_v_i_stopping_cond,
            epsilon_state=state_discretization_step,
            min_action_prob=min_action_prob,
            discretized_belief_states=self.oracle_strategy.discretized_belief_states,
            save_path=self.exp_type_path
        )

        oracle_strategy_basic_info_dict = self.oracle_strategy.generate_basic_info_dict()
        # optimistic_strategy_basic_info_dict = self.optimistic_algorithm_strategy.generate_basic_info_dict()

        # oracle_collected_samples = None
        # optimistic_alg_collected_samples = None
        # estimated_transition_matrices = np.zeros(shape=(num_experiments,
        #     num_episodes, self.num_states, self.num_actions,
        #     self.num_states))
        # frobenious_norm_error = np.zeros(shape=(num_experiments, num_episodes))

        for n in range(num_experiments):
            print("Experiment_n: " + str(n))

            if starting_episode_num != 0:
                oracle_starting_state, optimistic_starting_state = (
                    self.restore_infos(T_0=T_0, starting_episode_num=starting_episode_num,
                                   run_oracle=run_oracle,
                                   run_optimistic=run_optimistic,
                                   experiment_num=n))
            else:
                initial_state = np.random.multinomial(1, np.ones(shape=self.num_states) / self.num_states, 1)[
                 0].argmax()
                oracle_starting_state = initial_state
                optimistic_starting_state = initial_state

            if run_oracle is True:
                self.oracle_strategy.run(
                    T_0=T_0,
                    starting_episode_num=starting_episode_num,
                    num_episodes=num_episodes,
                    experiment_num=n,
                    initial_state=oracle_starting_state,
                )

            if run_optimistic is True:
                self.optimistic_algorithm_strategy.run(
                    T_0=T_0,
                    starting_episode_num=starting_episode_num,
                    num_episodes=num_episodes,
                    experiment_num=n,
                    initial_state=optimistic_starting_state,
                )

            if run_spectral is True:
                self.spectral_algorithm_strategy.run(
                    tau_1=tau_1,
                    tau_2=tau_2,
                    starting_episode_num=starting_episode_num,
                    num_episodes=num_episodes,
                    experiment_num=n,
                    initial_state=optimistic_starting_state,
                )

            if run_psrl is True:
                self.psrl_algorithm_strategy.run(
                    T_0=T_0,
                    starting_episode_num=starting_episode_num,
                    num_episodes=num_episodes,
                    experiment_num=n,
                    initial_state=optimistic_starting_state,
                )


        if not self.loaded_pomdp and self.save_pomdp_info:
            f = open(self.pomdp_dir_path + '/pomdp_info.json', 'w')
            json_file = json.dumps(pomdp_info_dict)
            f.write(json_file)
            f.close()

        if discretized_belief_states is None and self.save_basic_info:
            basic_info_path = f"/{state_discretization_step}stst_{min_action_prob}_minac"
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


    def restore_infos(self, T_0, starting_episode_num, run_oracle, run_optimistic, experiment_num):

        basic_info_path = f"/{self.state_discretization_step}stst_{self.min_action_prob}_minac/{T_0}_init"
        dir_to_read_path = self.exp_type_path + basic_info_path

        oracle_starting_state = None
        optimistic_starting_state = None
        if run_oracle is True:
            oracle_file_to_read_path = dir_to_read_path + f'/oracle_{starting_episode_num-1}Ep_{experiment_num}Exp.json'
            f = open(oracle_file_to_read_path)
            data = json.load(f)
            self.oracle_strategy.restore_infos(loaded_data=data)
            oracle_starting_state = data["starting_state"]

        if run_optimistic is True:
            optimistic_file_to_read_path = dir_to_read_path + f'/optimistic_{starting_episode_num - 1}Ep_{experiment_num}Exp.json'
            f = open(optimistic_file_to_read_path)
            data = json.load(f)
            self.optimistic_algorithm_strategy.restore_infos(loaded_data=data)
            optimistic_starting_state = data["starting_state"]

        return oracle_starting_state, optimistic_starting_state
