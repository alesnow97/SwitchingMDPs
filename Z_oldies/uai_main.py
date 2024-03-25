import utils
from Z_oldies.proposed_simulations.algorithms_comparison import CompareAlgo
from Z_oldies.proposed_simulations.algorithms_comparison_new import CompareAlgoNew
from Z_oldies.proposed_simulations.different_selected_arms import DifferentSelectedArms
from Z_oldies.proposed_simulations.estimation_error import EstimationErrorExp
from Z_oldies.proposed_simulations.estimation_error_all_arms import \
    EstimationErrorExpAllArms
from Z_oldies.proposed_simulations.estimation_error_hmm_rebuttal import \
    EstimationErrorExpRebuttal
from Z_oldies.proposed_simulations.estimation_error_v2 import EstimationErrorExpV2
from Z_oldies.proposed_simulations.estimation_error_v2_rebuttal import \
    EstimationErrorExpV2Rebuttal
from switchingBanditEnv import SwitchingBanditEnv


if __name__ == '__main__':
    num_states = 2
    num_actions = 4
    num_obs = 3
    total_horizon = 200000
    delta = 0.1

    compare_algo = False
    compare_algo_new = False
    estimation_error_exp = True

    hmm = False
    estimation_error_exp_rebuttal = False
    estimation_error_exp_all_arms = False
    estimation_error_expV2_rebuttal = False

    # if hmm:
    #     estimation_error_exp_rebuttal = True
    #     estimation_error_expV2_rebuttal = False
    # else:
    #     estimation_error_exp_rebuttal = False
    #     estimation_error_expV2_rebuttal = True

    estimation_error_expV2 = False
    different_selected_arms = False

    load_files = False
    save_results = False
    run_experiments = False
    save_bandit_info = False

    transition_from_file = None
    reference_matrix_from_file = None
    state_action_reward_matrix_from_file = None
    possible_rewards = None
    observation_multiplier = 20
    # transition_multiplier = 15
    transition_multiplier = 20
    num_experiments = 10

    bandit_to_load_path = 'experiments/3states_4actions_5obs/'
    bandit_num = 11

    experiments_samples = [900000, 1500000, 2100000, 3000000]
    num_bandits = 10
    num_arms_to_use = 3

    if load_files:
        num_states, num_actions, num_obs, \
            transition_from_file, reference_matrix_from_file, \
            state_action_reward_matrix_from_file, possible_rewards = \
            utils.load_files(bandit_to_load_path, bandit_num,
                             estimation_error_exp or estimation_error_expV2 or
                             estimation_error_exp_rebuttal or estimation_error_exp_all_arms)

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

    if compare_algo_new:
        num_particles = 300
        dirichlet_prior = (switching.transition_matrix * 170).astype(int)
        experiments = CompareAlgoNew(switching, num_experiments=num_experiments, sliding_window_size=1500, epsilon=0.05,
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
    if estimation_error_exp_all_arms:
        experiments = EstimationErrorExpAllArms(switching,
                                         num_experiments=num_experiments,
                                         save_results=save_results,
                                         save_bandit_info=save_bandit_info,
                                         loaded_bandit=load_files,
                                         bandit_num=bandit_num)
        experiments.run(starting_checkpoint=20000,
                        checkpoint_duration=20000,
                        num_checkpoints=50)
    if estimation_error_exp_rebuttal:
        for i in range(num_bandits):
            if load_files:
                num_states, num_actions, num_obs, \
                    transition_from_file, reference_matrix_from_file, \
                    state_action_reward_matrix_from_file, possible_rewards = \
                    utils.load_files(bandit_to_load_path, i,
                                     estimation_error_expV2 or estimation_error_exp_rebuttal)
            switching = SwitchingBanditEnv(num_states=num_states,
                                           num_actions=num_actions,
                                           num_obs=num_obs,
                                           transition_matrix=transition_from_file,
                                           state_action_reward_matrix=state_action_reward_matrix_from_file,
                                           reference_matrix=reference_matrix_from_file,
                                           possible_rewards=possible_rewards,
                                           transition_multiplier=transition_multiplier,
                                           observation_multiplier=observation_multiplier)
            experiments = EstimationErrorExpRebuttal(switching,
                                             save_results=save_results,
                                             save_bandit_info=save_bandit_info,
                                             loaded_bandit=load_files,
                                             bandit_num=i,
                                             dir_name=bandit_to_load_path,
                                             run_experiments=run_experiments)
            experiments.run(experiments_samples=experiments_samples)

    if estimation_error_expV2_rebuttal:
        for i in range(num_bandits):
            if load_files:
                num_states, num_actions, num_obs, \
                    transition_from_file, reference_matrix_from_file, \
                    state_action_reward_matrix_from_file, possible_rewards = \
                    utils.load_files(bandit_to_load_path, i,
                                     estimation_error_expV2_rebuttal or estimation_error_exp_rebuttal)
            switching = SwitchingBanditEnv(num_states=num_states,
                                           num_actions=num_actions,
                                           num_obs=num_obs,
                                           transition_matrix=transition_from_file,
                                           state_action_reward_matrix=state_action_reward_matrix_from_file,
                                           reference_matrix=reference_matrix_from_file,
                                           possible_rewards=possible_rewards,
                                           transition_multiplier=transition_multiplier,
                                           observation_multiplier=observation_multiplier)
            experiments = EstimationErrorExpV2Rebuttal(switching,
                                             num_experiments=num_experiments,
                                             save_results=save_results,
                                             save_bandit_info=save_bandit_info,
                                             loaded_bandit=load_files,
                                             dir_name=bandit_to_load_path,
                                             bandit_num=i)
            experiments.run(experiments_samples=experiments_samples, num_arms=num_arms_to_use)

    if estimation_error_expV2:
        experiments = EstimationErrorExpV2(switching,
                                         num_experiments=num_experiments,
                                         save_results=save_results,
                                         save_bandit_info=save_bandit_info,
                                         loaded_bandit=load_files,
                                         bandit_num=bandit_num)
        experiments.run(starting_checkpoint=20000,
                        checkpoint_duration=1000,
                        num_checkpoints=51)

    if different_selected_arms:
        experiments_selected_arms = DifferentSelectedArms(switching,
                                                          num_experiments=2,
                                                          sliding_window_size=100,
                                                          epsilon=0.5,
                                                          exp3S_gamma=0.01,
                                                          exp3S_limit=50,
                                                          exp3S_normalization_factor=100,
                                                          save_results=save_results)
        experiments_selected_arms.run(total_horizon=total_horizon,
                                      compute_regret_exploitation_horizon=False,
                                      save_results=save_results)

    # if save_files:
    #    utils.save_files(switching, experiments.store_dir_path)



