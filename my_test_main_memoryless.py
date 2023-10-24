import json

import numpy as np

import utils
from my_main_test.algorithms_comparison_memoryless import CompareAlgoMemoryless
from my_main_test.switchingBanditEnv_aistats_memoryless import \
    SwitchingBanditEnvAistats

np.random.seed(204)

if __name__ == '__main__':
    num_states = 4
    num_actions = 5
    num_obs = 6
    total_horizon = 6000002
    delta = 0.1

    compare_algo = False
    compare_algo_new = True
    compare_algo_movielens = True

    load_files = False
    save_results = True
    run_experiments = True
    save_bandit_info = False

    transition_from_file = None
    reference_matrix_from_file = None
    state_action_reward_matrix_from_file = None
    possible_rewards = None
    observation_multiplier = 80
    # transition_multiplier = 15
    transition_multiplier = 20
    num_experiments = 2

    bandit_to_load_path = 'experiments/3states_4actions_5obs/'
    bandit_num = 11

    experiments_samples = [9000, 1500000, 2100000, 3000000]
    num_bandits = 10
    num_arms_to_use = 5

    if load_files:
        num_states, num_actions, num_obs, \
            transition_from_file, reference_matrix_from_file, \
            state_action_reward_matrix_from_file, possible_rewards = \
            utils.load_files(bandit_to_load_path, bandit_num, False)

    switching = SwitchingBanditEnvAistats(num_states=num_states,
                                          num_actions=num_actions,
                                          num_obs=num_obs,
                                          transition_matrix=transition_from_file,
                                          state_action_reward_matrix=state_action_reward_matrix_from_file,
                                          arm_feature_representation=reference_matrix_from_file,
                                          possible_rewards=possible_rewards,
                                          transition_multiplier=transition_multiplier,
                                          observation_multiplier=observation_multiplier,
                                          memoryless=True)

    print("Finished")

    if compare_algo_new:
        num_particles = 300
        dirichlet_prior = (switching.transition_matrix * 1).astype(int)
        experiments = CompareAlgoMemoryless(switching,
                                  num_experiments=num_experiments,
                                  sliding_window_size=1500,
                                  epsilon=0.1,
                                  exp3S_gamma=0.01,
                                  exp3S_limit=20,
                                  exp3S_normalization_factor=100,
                                  save_results=save_results,
                                  save_bandit_info=save_bandit_info,
                                  loaded_bandit=load_files,
                                  bandit_num=bandit_num,
                                  num_particles=num_particles,
                                  lowest_prob=10**(-4),
                                  num_lowest_prob=num_particles/4,
                                  dirichlet_prior=dirichlet_prior)

        experiments.run(total_horizon=total_horizon,
                        compute_regret_exploitation_horizon=False)
