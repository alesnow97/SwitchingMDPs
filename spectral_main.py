
import numpy as np

from pomdp_env.POMDP import POMDP
from pomdp_env.pomdp_simulations_new import POMDPSimulationNew
from utils import load_pomdp, load_pomdp_basic_info


if __name__ == '__main__':

    run_settings = '1'

    if run_settings == '0':  # it corresponds to all new
        save_pomdp_info = True
        save_basic_info = True
        save_results = True
        to_load = False
        to_load_pomdp_basic_info = False
    elif run_settings == '1':  # it corresponds to same_pomdp_diff_discr
        save_pomdp_info = False
        save_basic_info = True
        save_results = True
        to_load = True
        to_load_pomdp_basic_info = False
        # elif run_settings == 'same_pomdp_same_discr':
    else:
        save_pomdp_info = False
        save_basic_info = False
        save_results = True
        to_load = True
        to_load_pomdp_basic_info = True

    # just for TESTING
    save_results = False

    # FARLO A MANOOO
    run_oracle = True
    run_optimistic = True

    num_states = 2
    num_actions = 2
    num_observations = 2
    num_experiments = 5

    # estimation error experiment
    num_samples_to_discard = 250
    num_samples_checkpoint = 20000
    num_checkpoints = 3

    # regret experiment
    ext_v_i_stopping_cond = 0.005
    state_discretization_step = 0.01

    non_normalized_min_transition_value = 0.05
    min_action_prob = 0.05
    delta = 0.9
    T_0 = 150
    starting_episode_num = 0
    num_episodes = 8

    tau_1 = 1500
    tau_2 = 3000

    pomdp_to_load_path = f"ICML_experiments/{num_states}states_{num_actions}actions_{num_observations}obs/"
    pomdp_num = 11

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
            starting_episode_num=starting_episode_num,
            tau_1=tau_1,
            tau_2=tau_2,
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
            starting_episode_num=starting_episode_num,
            tau_1=tau_1,
            tau_2=tau_2,
        )

    print("Ciao")

