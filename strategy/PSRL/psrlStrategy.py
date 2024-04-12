import json
import os
import time

import numpy as np

import utils
from policies.discretized_belief_based_policy import \
    DiscretizedBeliefBasedPolicy
from policies.psrl_discretized_belief_based_policy import \
    PSRLDiscretizedBeliefBasedPolicy
from strategy import strategy_helper


class PSRLStrategy:

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

        self.dirichlet_prior = np.ones((self.num_states, self.num_actions, self.num_states)) / num_states

        if discretized_belief_states is None:
            self.discretized_belief_states = utils.discretize_continuous_space(self.num_states, epsilon=epsilon_state)
            self.len_discretized_beliefs = self.discretized_belief_states.shape[0]

            start_time = time.time()
            self.real_belief_action_belief = strategy_helper.compute_belief_action_belief_matrix(
                num_actions=self.num_actions,
                num_obs=self.num_obs,
                discretized_belief_states=self.discretized_belief_states,
                len_discretized_beliefs=self.len_discretized_beliefs,
                state_action_transition_matrix=self.pomdp.state_action_transition_matrix,
                state_action_observation_matrix=self.pomdp.state_action_observation_matrix
            )

            end_time = time.time()
            compute_belief_action_list_time = end_time - start_time
            print(f"Compute_belief_action_list time is {compute_belief_action_list_time}")

            start_time = time.time()

            self.real_optimal_belief_action_mapping = strategy_helper.compute_optimal_POMDP_policy(
                num_actions=self.num_actions,
                discretized_belief_states=self.discretized_belief_states,
                len_discretized_beliefs=self.len_discretized_beliefs,
                ext_v_i_stopping_cond=self.ext_v_i_stopping_cond,
                min_action_prob=self.min_action_prob,
                state_action_reward=self.pomdp.state_action_reward,
                belief_action_belief_matrix=self.real_belief_action_belief
            )
            end_time = time.time()
            optimal_POMDP_policy_time = end_time - start_time
            print(f"Optimal_POMDP_policy_time is {optimal_POMDP_policy_time}")


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

        self.psrl_policy = DiscretizedBeliefBasedPolicy(
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
            "real_belief_action_belief": self.real_belief_action_belief,  #.tolist(),
            "real_optimal_belief_action_mapping": self.real_optimal_belief_action_mapping.tolist(),
            "initial_discretized_belief": self.initial_discretized_belief.tolist(),
            "initial_discretized_belief_index": self.initial_discretized_belief_index,
            "ext_v_i_stopping_cond": self.ext_v_i_stopping_cond,
            "epsilon_state": self.epsilon_state,
            "min_action_prob": self.min_action_prob
        }


    def run(self, T_0, starting_episode_num, num_episodes, experiment_num,
            initial_state):

        # self.estimated_action_state_dist_per_episode = np.zeros(shape=(
        #     num_episodes, self.num_actions, self.num_actions,
        #     self.num_states, self.num_states))

        if starting_episode_num == 0:
            self.init_policy()

        # self.estimated_transition_matrix_per_episode = np.zeros(shape=(
        #     num_episodes, self.num_states, self.num_actions,
        #     self.num_states))
        #
        # self.error_frobenious_norm_per_episode = np.empty(shape=num_episodes)

        # finché non è finita
        T_max = 0

        self.collected_samples = None
        self.frobenious_norms = []

        for i in range(starting_episode_num,
                       starting_episode_num + num_episodes):

            num_samples_to_discard = int(np.log(T_0 * 2 ** i))
            num_samples_in_episode = int(T_0 * 2 ** i)
            T_max += (num_samples_to_discard + num_samples_in_episode)


        episode_num = starting_episode_num
        t = 1
        t_k = 0

        # m_t = np.zeros(shape=(self.num_states, self.num_actions))
        # m_tk = np.zeros(shape=(self.num_states, self.num_actions))
        m_t = np.zeros(shape=(self.num_actions))

        first_sample = True
        first_state = initial_state

        while t <= 2 * T_max:       # it is multiplied by 2 since each sample is a couple

            T_k_1 = t - t_k
            t_k = t
            m_tk = m_t.copy()

            # sample here the transition probability
            trans_matrix_frobenious_norm = self.sample_transition_matrix()
            self.frobenious_norms.append(trans_matrix_frobenious_norm)

            # compute here the optimal policy
            # compute optimistic mdp
            optimistic_transition_matrix_mdp, optimistic_policy_mdp = (
                strategy_helper.compute_optimistic_MDP(
                    num_states=self.num_states,
                    num_actions=self.num_actions,
                    min_action_prob=self.min_action_prob,
                    state_action_transition_matrix=self.estimated_transition_matrix,
                    ext_v_i_stopping_cond=self.ext_v_i_stopping_cond,
                    state_action_reward=self.pomdp.state_action_reward,
                    confidence_bound=0,
                    min_transition_value=self.pomdp.min_transition_value
                ))

            start_time = time.time()
            # compute belief action belief matrix from the optimistic mdp
            optimistic_belief_action_belief_matrix = (
                strategy_helper.compute_belief_action_belief_matrix(
                    num_actions=self.num_actions,
                    num_obs=self.num_obs,
                    discretized_belief_states=self.discretized_belief_states,
                    len_discretized_beliefs=self.len_discretized_beliefs,
                    state_action_transition_matrix=optimistic_transition_matrix_mdp,
                    state_action_observation_matrix=self.pomdp.state_action_observation_matrix
                ))
            end_time = time.time()
            compute_belief_action_list_time = end_time - start_time
            print(
                f"Compute_belief_action_list time is {compute_belief_action_list_time}")

            start_time = time.time()
            optimistic_belief_action_mapping = strategy_helper.compute_optimal_POMDP_policy(
                num_actions=self.num_actions,
                discretized_belief_states=self.discretized_belief_states,
                len_discretized_beliefs=self.len_discretized_beliefs,
                ext_v_i_stopping_cond=self.ext_v_i_stopping_cond,
                min_action_prob=self.min_action_prob,
                state_action_reward=self.pomdp.state_action_reward,
                belief_action_belief_matrix=optimistic_belief_action_belief_matrix
            )
            end_time = time.time()
            optimal_POMDP_policy_time = end_time - start_time
            print(f"Optimal_POMDP_policy_time is {optimal_POMDP_policy_time}")

            self.policy.update_policy_infos(
                state_action_transition_matrix=optimistic_transition_matrix_mdp,
                belief_action_dist_mapping=optimistic_belief_action_mapping
            )

            num_last_samples_for_belief_update = 50
            self.policy.update_belief_from_samples(
                action_obs_samples=self.collected_samples[
                                   -num_last_samples_for_belief_update:]
            )


            episode_collected_samples = []

            while t <= t_k + T_k_1 and np.all(m_t <= 2 * m_tk):

                first_action = self.policy.choose_action()
                first_obs = np.random.multinomial(
                    n=1, pvals=self.pomdp.state_action_observation_matrix[
                        first_state, first_action],
                    size=1)[0].argmax()

                second_state = np.random.multinomial(
                    n=1, pvals=self.pomdp.state_action_transition_matrix[
                        first_state, first_action], size=1)[
                    0].argmax()

                self.policy.update(first_action, first_obs)
                episode_collected_samples.append((first_action,
                                                  first_obs,
                                                  self.pomdp.possible_rewards[first_obs]))
                # self.collected_samples.append((first_action, first_obs))

                if first_sample is False:

                    # update here dirichlet prior
                    # qui serve la belief allo stato precedente
                    self.update_dirichlet_prior(episode_collected_samples[-2:])


                if first_sample is True:
                    if len(episode_collected_samples) > 1:
                        first_sample = False

                first_state = second_state
                t = t + 1


            if self.collected_samples is not None:
                self.collected_samples = np.vstack([self.collected_samples, np.array(episode_collected_samples)])
            else:
                self.collected_samples = np.array(episode_collected_samples)




            # episode_num = episode_num + 1

        self.save_results(
            T_0=T_0,
            experiment_num=experiment_num,
            starting_state=initial_state,
            frobenious_norm=self.frobenious_norms
        )




    def update_dirichlet_prior(self, last_samples: list):

        first_action, first_obs = last_samples[0]
        second_action, second_obs = last_samples[1]

        first_belief = self.policy.previous_discretized_belief

        modified_belief = first_belief * self.pomdp.state_action_observation_matrix[:, first_action, first_obs]

        state_action_trans = self.estimated_transition_matrix[:, first_action, :]

        scaled_state_action_trans = modified_belief[:, None] * state_action_trans

        second_action_obs_prob = self.pomdp.state_action_observation_matrix[:, second_action, second_obs]

        final_update = second_action_obs_prob[:, None] * scaled_state_action_trans

        final_update = final_update / final_update.sum()

        self.dirichlet_prior[:, first_action, :] = final_update


    def sample_transition_matrix(self):

        self.estimated_transition_matrix = np.empty(shape=(self.num_states, self.num_actions, self.num_states))

        for state in range(self.num_states):
            for action in range(self.num_actions):
                current_dirichlet_prior = self.dirichlet_prior[state, action]
                self.estimated_transition_matrix[state, action] = np.random.dirichlet(current_dirichlet_prior)

        print("The new transition matrix has been sampled!")

        distance_matrix = np.absolute(self.pomdp.state_action_transition_matrix.reshape(-1) -
            self.estimated_transition_matrix.reshape(-1))

        frobenious_norm = np.sqrt(np.sum(distance_matrix**2))

        return frobenious_norm


    def save_results(self,
                     T_0,       # this quantity is only used to save the results in the right directory
                     experiment_num,
                     starting_state,
                     frobenious_norm
                     ):

        if isinstance(self.policy.discretized_belief_index, int):
            index_to_store = self.policy.discretized_belief_index
        else:
            index_to_store = self.policy.discretized_belief_index.tolist()

        result_dict = {
            "starting_state": starting_state.tolist(),
            "discretized_belief": self.policy.discretized_belief.tolist(),
            "discretized_belief_index": index_to_store,
            "frobenious_norm": frobenious_norm,
            "collected_samples": self.collected_samples.tolist()
        }

        # basic_info_path = f"/{self.epsilon_state}stst_{self.min_action_prob}_minac/{T_0}_init"
        # dir_to_create_path = self.save_path + basic_info_path
        # if not os.path.exists(dir_to_create_path):
        #     os.mkdir(dir_to_create_path)
        # f = open(
        #     dir_to_create_path + f'/optimistic_{episode_num}Ep_{experiment_num}Exp.json',
        #     'w')
        # json_file = json.dumps(result_dict)
        # f.write(json_file)
        # f.close()
        # print(f"Optimistic Results of episode {episode_num} and experiment {experiment_num} have been saved")

        basic_info_path = f"/{self.epsilon_state}stst_{self.min_action_prob}_minac/{T_0}_init"
        dir_to_create_path = self.save_path + basic_info_path
        if not os.path.exists(dir_to_create_path):
            os.mkdir(dir_to_create_path)
        f = open(
            dir_to_create_path + f'/psrl_{experiment_num}Exp.json',
            'w')
        json_file = json.dumps(result_dict)
        f.write(json_file)
        f.close()
        print(f"PSRL Results of and experiment {experiment_num} have been saved")


    # TODO se serve inserire il metodo che fa restore infos

    def init_policy(self):

        # used policy
        self.policy = PSRLDiscretizedBeliefBasedPolicy(
            num_states=self.num_states,
            num_actions=self.num_actions,
            num_obs=self.num_obs,
            initial_discretized_belief=None,
            initial_discretized_belief_index=None,
            discretized_beliefs=self.discretized_belief_states,
            estimated_state_action_transition_matrix=None,
            belief_action_dist_mapping=None,
            state_action_observation_matrix=self.pomdp.state_action_observation_matrix,
            no_info=True
        )

    def generate_basic_info_dict(self):

        experiment_basic_info = {
            "discretized_belief_states": self.discretized_belief_states.tolist(),
            "ext_v_i_stopping_cond": self.ext_v_i_stopping_cond,
            "epsilon_state": self.epsilon_state,
            "min_action_prob": self.min_action_prob
        }

        return experiment_basic_info



