import numpy as np

from Z_SwitchingMDP.environment.switchingMDPs import SwitchingMDPs
from policies.belief_based_policy import BeliefBasedPolicy
from policies.policy import Policy
from policies.state_based_policy import StateBasedPolicy
from pomdp_env import POMDP
from strategy.mixtureStrategy import MixtureStrategy
from strategy.weightedMeanStrategy import WeightedMeanStrategy
from utils import compute_min_svd

np.random.seed(200)


class POMDPSimulation:

    def __init__(self,
                 pomdp: POMDP
                 ):
        self.pomdp = pomdp
        self.num_states = self.pomdp.num_states
        self.num_actions = self.pomdp.num_actions
        self.num_obs = self.pomdp.num_obs


    def run(self, num_experiments: int, num_samples_to_discard: int,
            num_samples_per_estimate: int, num_estimates: int):

        # bandit_info_dict = {
        #     'transition_matrix': self.switching_env.transition_matrix.tolist(),
        #     # 'reference_matrix': self.switching_env.arm.tolist(),
        #     'state_action_reward_matrix': self.switching_env.state_action_reward_matrix.tolist(),
        #     'possible_rewards': self.switching_env.possible_rewards.tolist(),
        #     'num_states': self.switching_env.num_states,
        #     'num_actions': self.switching_env.num_actions,
        #     'num_obs': self.switching_env.num_obs,
        #     'observation_multiplier': self.switching_env.observation_multiplier,
        #     'transition_multiplier': self.switching_env.transition_multiplier
        # }

        # result_dict = {'total_horizon': total_horizon,
        #                'epsilon': self.epsilon,
        #                'sliding_window_size': self.sliding_window_size,
        #                'exp3S_gamma': self.exp3S_gamma,
        #                'exp3S_normalization_factor': self.exp3S_normalization_factor,
        #                'exp3S_limit': self.exp3S_limit,
        #                'num_experiments': self.num_experiments,
        #                'algorithms_to_use': algorithms_to_use, 'rewards': {}}

        # mode_state_action_rew_list = np.empty(
        #     shape=(num_experiments, horizon, 4), dtype=int)


        # state_action_reward_matrix = self.switching_env.state_action_reward_matrix
        # possible_rewards = self.switching_env.possible_rewards

        for n in range(num_experiments):
            print("experiment_n: " + str(n))

            initial_state = np.random.random_integers(low=0, high=self.num_states-1)

            # sets of used policies
            self.init_policies()
            # self.sample_reuse_strategy = MixtureStrategy(policy=self.policy_reuse,
            #                                              num_states=self.num_states,
            #                                              num_actions=self.num_actions,
            #                                              num_obs=self.num_obs,
            #                                              pomdp=self.pomdp,
            #                                              sample_reuse=True)

            self.sample_reuse_strategy = WeightedMeanStrategy(policy=self.policy_reuse,
                                                         num_states=self.num_states,
                                                         num_actions=self.num_actions,
                                                         num_obs=self.num_obs,
                                                         pomdp=self.pomdp,
                                                         sample_reuse=True)

            self.sample_reuse_strategy.run(num_samples_to_discard=num_samples_to_discard,
                                           num_samples_per_estimate=num_samples_per_estimate,
                                           num_estimates=num_estimates,
                                           initial_state=initial_state)

            # self.sample_no_reuse_strategy = Strategy(
            #     policy=self.policy_no_reuse,
            #     num_states=self.num_states,
            #     num_actions=self.num_actions,
            #     num_obs=self.num_obs,
            #     pomdp=self.pomdp)


    def init_policies(self):

        self.policy_no_reuse = BeliefBasedPolicy(
            self.num_states, self.num_actions, self.num_obs,
            self.pomdp.state_action_transition_matrix,
            self.pomdp.state_action_observation_matrix,
            self.pomdp.possible_rewards
        )

        self.policy_reuse = BeliefBasedPolicy(
            self.num_states, self.num_actions, self.num_obs,
            self.pomdp.state_action_transition_matrix,
            self.pomdp.state_action_observation_matrix,
            self.pomdp.possible_rewards
        )
