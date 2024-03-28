import json
import os

import numpy as np

import utils

from policies.discretized_belief_based_policy import \
    DiscretizedBeliefBasedPolicy
from strategy import strategy_helper


class OracleStrategy:

    def __init__(self,
                 num_states,
                 num_actions,
                 num_obs,
                 pomdp,
                 ext_v_i_stopping_cond=0.02,
                 epsilon_state=0.2,
                 min_action_prob=0.1,
                 discretized_belief_states=None,
                 real_belief_action_belief=None,
                 real_optimal_belief_action_mapping=None,
                 initial_discretized_belief=None,
                 initial_discretized_belief_index=None,
                 save_path=None,
                 ):

        self.num_states = num_states
        self.num_actions = num_actions
        self.num_obs = num_obs
        self.pomdp = pomdp
        self.ext_v_i_stopping_cond = ext_v_i_stopping_cond
        self.epsilon_state = epsilon_state
        self.min_action_prob = min_action_prob

        self.save_path = save_path

        if discretized_belief_states is None:
            self.discretized_belief_states = utils.discretize_continuous_space(self.num_states, epsilon=epsilon_state)
            self.len_discretized_beliefs = self.discretized_belief_states.shape[0]

            self.real_belief_action_belief = strategy_helper.compute_belief_action_belief_matrix(
                num_actions=self.num_actions,
                num_obs=self.num_obs,
                discretized_belief_states=self.discretized_belief_states,
                len_discretized_beliefs=self.len_discretized_beliefs,
                state_action_transition_matrix=self.pomdp.state_action_transition_matrix,
                state_action_observation_matrix=self.pomdp.state_action_observation_matrix
            )

            self.real_optimal_belief_action_mapping = strategy_helper.compute_optimal_POMDP_policy(
                num_actions=self.num_actions,
                discretized_belief_states=self.discretized_belief_states,
                len_discretized_beliefs=self.len_discretized_beliefs,
                ext_v_i_stopping_cond=self.ext_v_i_stopping_cond,
                min_action_prob=self.min_action_prob,
                state_action_reward=self.pomdp.state_action_reward,
                belief_action_belief_matrix=self.real_belief_action_belief
            )

            uniform_initial_belief = np.ones(shape=self.num_states) / self.num_states
            self.initial_discretized_belief, self.initial_discretized_belief_index = utils.find_closest_discretized_belief(
                self.discretized_belief_states, uniform_initial_belief
            )
        else:
            # data come from a loaded file
            self.discretized_belief_states = discretized_belief_states
            self.len_discretized_beliefs = self.discretized_belief_states.shape[0]
            self.real_belief_action_belief = real_belief_action_belief
            self.real_optimal_belief_action_mapping = real_optimal_belief_action_mapping
            self.initial_discretized_belief = initial_discretized_belief
            self.initial_discretized_belief_index = initial_discretized_belief_index

        self.oracle_policy = DiscretizedBeliefBasedPolicy(
            num_states=self.num_states,
            num_actions=self.num_actions,
            num_obs=self.num_obs,
            initial_discretized_belief=self.initial_discretized_belief,
            initial_discretized_belief_index=self.initial_discretized_belief_index,
            discretized_beliefs=self.discretized_belief_states,
            estimated_state_action_transition_matrix=self.pomdp.state_action_transition_matrix,
            belief_action_dist_mapping=self.real_optimal_belief_action_mapping,
            state_action_observation_matrix=self.pomdp.state_action_observation_matrix,
            no_info=False
        )

        self.experiment_info = {
            "discretized_belief_states": self.discretized_belief_states.tolist(),
            "real_belief_action_belief": self.real_belief_action_belief.tolist(),
            "real_optimal_belief_action_mapping": self.real_optimal_belief_action_mapping.tolist(),
            "initial_discretized_belief": self.initial_discretized_belief.tolist(),
            "initial_discretized_belief_index": self.initial_discretized_belief_index,
            "ext_v_i_stopping_cond": self.ext_v_i_stopping_cond,
            "epsilon_state": self.epsilon_state,
            "min_action_prob": self.min_action_prob
        }


    def run(self, T_0, starting_episode_num, num_episodes, experiment_num, initial_state):
        current_state = initial_state

        # self.estimated_action_state_dist_per_episode = np.zeros(shape=(
        #     num_episodes, self.num_actions, self.num_actions,
        #     self.num_states, self.num_states))

        self.collected_samples = None

        for i in range(starting_episode_num, starting_episode_num + num_episodes):

            episode_collected_samples, last_state = self.collect_samples_in_episode(
                starting_state=current_state,
                T_0=T_0,
                episode_num=i
            )

            current_state = last_state

            self.collected_samples = episode_collected_samples
            self.save_results(T_0=T_0, episode_num=i, experiment_num=experiment_num,
                              starting_state=current_state)

            # if self.collected_samples is None:
            #     self.collected_samples = episode_collected_samples
            # else:
            #     self.collected_samples = np.vstack([self.collected_samples, episode_collected_samples])

        # return self.collected_samples


    def collect_samples_in_episode(self, starting_state, T_0, episode_num):
        # for convenience these numbers are even
        num_samples_to_discard = int(np.log(T_0 * 2**episode_num))
        num_samples_in_episode = int(T_0 * 2**episode_num)

        self.full_obs_count_vec = np.zeros(
            shape=(self.num_states, self.num_states,
                   self.num_actions, self.num_actions,
                   self.num_obs, self.num_obs))

        self.part_obs_count_vec = np.zeros(
            shape=(self.num_actions, self.num_actions,
                   self.num_obs, self.num_obs))

        num_total_samples = num_samples_to_discard + num_samples_in_episode
        first_state = starting_state

        episode_collected_samples = np.zeros(shape=(2*num_total_samples, 3))

        for sample_num in range(num_total_samples):

            first_action = self.oracle_policy.choose_action()
            first_obs = np.random.multinomial(
                n=1, pvals=self.pomdp.state_action_observation_matrix[
                    first_state, first_action],
                size=1)[0].argmax()

            second_state = np.random.multinomial(
                n=1, pvals=self.pomdp.state_action_transition_matrix[
                    first_state, first_action], size=1)[
                0].argmax()

            self.oracle_policy.update(first_action, first_obs)
            episode_collected_samples[2*sample_num] = [first_action, first_obs, self.pomdp.possible_rewards[first_obs]]
            #self.collected_samples.append((first_action, first_obs))

            second_action = self.oracle_policy.choose_action()
            second_obs = np.random.multinomial(
                n=1, pvals=self.pomdp.state_action_observation_matrix[
                    second_state, second_action],
                size=1)[0].argmax()

            next_state = np.random.multinomial(
                n=1, pvals=self.pomdp.state_action_transition_matrix[
                    second_state, second_action], size=1)[
                0].argmax()

            self.oracle_policy.update(second_action, second_obs)
            episode_collected_samples[2 * sample_num + 1] = [second_action,
                                                             second_obs,
                                                             self.pomdp.possible_rewards[second_obs]]
            # self.collected_samples.append((second_action, second_obs))

            if sample_num % 1000 == 0:
                print(sample_num)

            first_state = next_state

        return episode_collected_samples, first_state


    def save_results(self, T_0, episode_num, experiment_num, starting_state):

        if isinstance(self.oracle_policy.discretized_belief_index, int):
            index_to_store = self.oracle_policy.discretized_belief_index
        else:
            index_to_store = self.oracle_policy.discretized_belief_index.tolist()

        result_dict = {
            "starting_state": starting_state.tolist(),
            "discretized_belief": self.oracle_policy.discretized_belief.tolist(),
            "discretized_belief_index": index_to_store,
            "collected_samples": self.collected_samples.tolist()
        }

        exp_path = self.save_path + f"/{self.epsilon_state}stst_{self.min_action_prob}_minac"
        if not os.path.exists(exp_path):
            os.mkdir(exp_path)
        dir_to_create_path = exp_path + f"/{T_0}_init"
        #
        # basic_info_path = f"/{self.epsilon_state}stst_{self.epsilon_action}acst_{self.min_action_prob}_minac/{T_0}_init"
        # dir_to_create_path = self.save_path + basic_info_path
        if not os.path.exists(dir_to_create_path):
            os.mkdir(dir_to_create_path)
        f = open(
            dir_to_create_path + f'/oracle_{episode_num}Ep_{experiment_num}Exp.json',
            'w')
        json_file = json.dumps(result_dict)
        f.write(json_file)
        f.close()
        print(f"Oracle Results of episode {episode_num} and experiment {experiment_num} have been saved")

    def restore_infos(self,
                      loaded_data):
        self.oracle_policy.discretized_belief = np.array(loaded_data["discretized_belief"])
        self.oracle_policy.discretized_belief_index = loaded_data["discretized_belief_index"]


    def generate_basic_info_dict(self):

        if isinstance(self.initial_discretized_belief_index, int):
            index_to_store = self.initial_discretized_belief_index
        else:
            index_to_store = self.initial_discretized_belief_index.tolist()

        experiment_basic_info = {
            "discretized_belief_states": self.discretized_belief_states.tolist(),
            "real_belief_action_belief": self.real_belief_action_belief.tolist(),
            "real_optimal_belief_action_mapping": self.real_optimal_belief_action_mapping.tolist(),
            "initial_discretized_belief": self.initial_discretized_belief.tolist(),
            "initial_discretized_belief_index": index_to_store,
            "ext_v_i_stopping_cond": self.ext_v_i_stopping_cond,
            "epsilon_state": self.epsilon_state,
            "min_action_prob": self.min_action_prob
        }

        return experiment_basic_info




