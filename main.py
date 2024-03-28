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
parser.add_argument('--num_episodes', help='Number of episodes', type=int, default=None)
parser.add_argument('--T_0', help='First episode length', type=int, default=None)
parser.add_argument('--run_settings', help='Run settings', type=str, default='all_new',
                    choices=['all_new', 'same_pomdp_diff_discr', 'same_pomdp_same_discr'])
parser.add_argument('--non_normalized_min_transition_value', help='Non normalized min_transition_value', type=float, default=None)
parser.add_argument('--min_action_prob', help='Min action probability', type=float, default=None)
parser.add_argument('--delta', help='Confidence level', type=float, default=0.9)
parser.add_argument('--starting_episode_num', help='starting_episode_num', type=int, default=0)
parser.add_argument('--pomdp_num', help='Id of the POMDP to be restored', type=int, default=None)
parser.add_argument('--ext_v_i_stopping_cond', help='Stopping Condition', type=float, default=0.0005)
parser.add_argument('--state_discretization_step', help='State Discretization step', type=float, default=None)

args = parser.parse_args()

if __name__ == '__main__':

    # run settings
    # all_new = True
    # same_pomdp_diff_discr = False
    # same_pomdp_same_discr = False

    run_settings = args.run_settings
    print("Ciaoooo")

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
    #elif run_settings == 'same_pomdp_same_discr':
    else:
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
    # num_samples_to_discard = 250
    # num_samples_checkpoint = 20000
    # num_checkpoints = 5

    # regret experiment
    ext_v_i_stopping_cond = args.ext_v_i_stopping_cond
    state_discretization_step = args.state_discretization_step

    non_normalized_min_transition_value = args.non_normalized_min_transition_value
    min_action_prob = args.min_action_prob
    delta = args.delta
    T_0 = args.T_0
    starting_episode_num = args.starting_episode_num
    num_episodes = args.num_episodes

    pomdp_to_load_path = f"ICML_experiments/{num_states}states_{num_actions}actions_{num_observations}obs/"
    pomdp_num = args.pomdp_num

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

