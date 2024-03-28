import argparse

import numpy as np

from pomdp_env.POMDP import POMDP
from pomdp_env.pomdp_simulations_new import POMDPSimulationNew
from utils import load_pomdp, load_pomdp_basic_info

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--num_states', help='Number of states', type=int, default=None)
parser.add_argument('--num_actions', help='Number of actions', type=int, default=None)
parser.add_argument('--num_obs', help='Number of observations', type=int, default=None)
parser.add_argument('--num_experiments', help='Number of experiments', type=int, default=None)
parser.add_argument('--run_settings', help='Run settings', type=str, default='all_new',
                    choices=['all_new', 'same_pomdp_diff_discr', 'same_pomdp_same_discr'])




parser.add_argument('--command', help='Command to execute.', type=str, default='launch', choices=['launch', 'view', 'stop'])
# Experiment selection
parser.add_argument('--name', help='Name of the experiment', type=str, default=None)
parser.add_argument('--dir', help='Directory from which to load the experiment (to launch).', type=str, default=None)
# Env
parser.add_argument('--condaenv', help='Conda environment to activate.', type=str, default=None)
parser.add_argument('--pythonv', help='Python version to use', type=str, default='python3')
parser.add_argument('--pythonpath', help='Pythonpath to use for script.', type=str, default=None)
parser.add_argument('--cuda_devices', help='CUDA visible devices.', type=str, default='')
# Sacred
parser.add_argument('--sacred', action='store_true', default=False, help='Enable sacred.')
parser.add_argument('--sacred_dir', help='Dir used by sacred to log.', type=str, default=None)
parser.add_argument('--sacred_slack', help='Config file for slack.', type=str, default=None)
parser.add_argument('--dirty', action='store_true', default=False, help='Enable sacred dirty running.')
args = parser.parse_args()

if __name__ == '__main__':

    # run settings
    # all_new = True
    # same_pomdp_diff_discr = False
    # same_pomdp_same_discr = False

    run_settings = args.run_settings

    if run_settings == 'all_new':
        save_pomdp_info = True
        save_basic_info = True
        save_results = True
        to_load = False
        to_load_pomdp_basic_info = False
    elif run_settings == 'same_pomdp_diff_discr':
        save_pomdp_info = False
        save_basic_info = True
        save_results = True
        to_load = True
        to_load_pomdp_basic_info = False
    elif run_settings == 'same_pomdp_same_discr':
        save_pomdp_info = False
        save_basic_info = False
        save_results = True
        to_load = True
        to_load_pomdp_basic_info = True

    # FARLO A MANOOO
    run_oracle = True
    run_optimistic = True

    num_states = args.num_states
    num_actions = args.num_actions
    num_observations = args.num_obs
    num_experiments = args.num_experiments

    # estimation error experiment
    num_samples_to_discard = 250
    num_samples_checkpoint = 20000
    num_checkpoints = 5

    # regret experiment
    ext_v_i_stopping_cond = 0.005
    state_discretization_step = 0.04

    non_normalized_min_transition_value = 0.05
    min_action_prob = 0.02
    delta = 0.9
    T_0 = 5000
    starting_episode_num = 0
    num_episodes = 2

    pomdp_to_load_path = f"ICML_experiments/{num_states}states_{num_actions}actions_{num_observations}obs/"
    pomdp_num = 1

    if to_load:
        pomdp = load_pomdp(pomdp_to_load_path, pomdp_num)
    else:
        possible_rewards = np.random.permutation(
            np.linspace(start=0.0, stop=1.0, num=num_observations))
        pomdp = POMDP(
            num_states=num_states,
            num_actions=num_actions,
            num_observations=num_observations,
            possible_rewards=possible_rewards,
            real_min_transition_value=None,
            non_normalized_min_transition_value=non_normalized_min_transition_value,
            state_action_transition_matrix=None,
            state_action_observation_matrix=None,
            observation_multiplier=10
        )

    simulation = POMDPSimulationNew(pomdp,
                                    loaded_pomdp=to_load,
                                    pomdp_num=pomdp_num,
                                    save_pomdp_info=save_pomdp_info,
                                    save_basic_info=save_basic_info,
                                    save_results=save_results
                                    )

    if to_load_pomdp_basic_info:
        (discretized_belief_states,
         real_belief_action_belief, real_optimal_belief_action_mapping,
         initial_discretized_belief, initial_discretized_belief_index) = (
            load_pomdp_basic_info(
            pomdp_to_load_path=pomdp_to_load_path,
            pomdp_num=pomdp_num,
            state_discretization_step=state_discretization_step,
            min_action_prob=min_action_prob
        ))
        simulation.run_regret_experiment(
            num_experiments=num_experiments,
            T_0=T_0,
            num_episodes=num_episodes,
            ext_v_i_stopping_cond=ext_v_i_stopping_cond,
            state_discretization_step=state_discretization_step,
            min_action_prob=min_action_prob,
            delta=delta,
            discretized_belief_states=discretized_belief_states,
            real_belief_action_belief=real_belief_action_belief,
            real_optimal_belief_action_mapping=real_optimal_belief_action_mapping,
            initial_discretized_belief=initial_discretized_belief,
            initial_discretized_belief_index=initial_discretized_belief_index,
            run_oracle=run_oracle,
            run_optimistic=run_optimistic,
            starting_episode_num=starting_episode_num
        )
    else:
        simulation.run_regret_experiment(
            num_experiments=num_experiments,
            T_0=T_0,
            num_episodes=num_episodes,
            ext_v_i_stopping_cond=ext_v_i_stopping_cond,
            state_discretization_step=state_discretization_step,
            min_action_prob=min_action_prob,
            delta=delta,
            run_oracle=run_oracle,
            run_optimistic=run_optimistic,
            starting_episode_num=starting_episode_num
        )

    print("Ciao")
