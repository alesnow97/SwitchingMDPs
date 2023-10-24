import utils
from proposed_simulations.algorithms_comparison import CompareAlgo
from proposed_simulations.estimation_error import EstimationErrorExp
from switchingBanditEnv import SwitchingBanditEnv


if __name__ == '__main__':
    num_states = 3
    num_actions = 4
    num_obs = 5
    total_horizon = 50000
    delta = 0.1

    compare_algo = False
    estimation_error_exp = False
    estimation_error_expV2 = False
    different_selected_arms = False
    optimistic = True

    load_files = False
    save_results = False
    save_bandit_info = False

    transition_from_file = None
    reference_matrix_from_file = None
    state_action_reward_matrix_from_file = None
    possible_rewards = None
    observation_multiplier = 20
    # transition_multiplier = 15
    transition_multiplier = 10
    num_experiments = 10

    bandit_to_load_path = 'experiments_estimation_error_v2/5states_8actions_10obs'
    bandit_num = 0

    if load_files:
        num_states, num_actions, num_obs, \
            transition_from_file, reference_matrix_from_file, \
            state_action_reward_matrix_from_file, possible_rewards = \
            utils.load_files(bandit_to_load_path, bandit_num, estimation_error_expV2)

    switching = SwitchingBanditEnv(num_states=num_states,
                                   num_actions=num_actions,
                                   num_obs=num_obs,
                                   transition_matrix=transition_from_file,
                                   state_action_reward_matrix=state_action_reward_matrix_from_file,
                                   reference_matrix=reference_matrix_from_file,
                                   possible_rewards=possible_rewards,
                                   transition_multiplier=transition_multiplier,
                                   observation_multiplier=observation_multiplier)

    exploration_horizon = utils.find_optimal_exploration_length(switching, total_horizon, delta)
    print(exploration_horizon)

    if compare_algo:
        experiments = CompareAlgo(switching, num_experiments=num_experiments, sliding_window_size=1500, epsilon=0.05,
                                  exp3S_gamma=0.01,
                                  exp3S_limit=20,
                                  exp3S_normalization_factor=100,
                                  save_results=save_results,
                                  save_bandit_info=save_bandit_info,
                                  loaded_bandit=load_files,
                                  bandit_num=bandit_num)
        experiments.run(total_horizon=total_horizon,
                        exploration_horizon=exploration_horizon,
                        compute_regret_exploitation_horizon=False)

    if estimation_error_exp:
        experiments = EstimationErrorExp(switching,
                                         num_experiments=num_experiments,
                                         save_results=save_results,
                                         save_bandit_info=save_bandit_info,
                                         loaded_bandit=load_files,
                                         bandit_num=bandit_num)
        experiments.run(starting_checkpoint=20000,
                        checkpoint_duration=1000,
                        num_checkpoints=51)

    experiments = EstimationErrorExp(switching,
                                     num_experiments=num_experiments,
                                     save_results=save_results,
                                     save_bandit_info=save_bandit_info,
                                     loaded_bandit=load_files,
                                     bandit_num=bandit_num)
    experiments.run(starting_checkpoint=20000,
                    checkpoint_duration=1000,
                    num_checkpoints=51)



    # if save_files:
    #    utils.save_files(switching, experiments.store_dir_path)



