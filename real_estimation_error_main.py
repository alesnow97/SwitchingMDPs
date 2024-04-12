
import numpy as np

from pomdp_env.POMDP import POMDP
from pomdp_env.pomdp_simulations_new import POMDPSimulationNew
from utils import load_pomdp, load_pomdp_basic_info


if __name__ == '__main__':

    run_settings = '0'

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

    # FARLO A MANOOO
    run_oracle = True
    run_optimistic = True

    num_states = 10
    num_actions = 6
    num_observations = 15
    num_experiments = 2

    # estimation error experiment
    num_samples_to_discard = 250
    num_samples_checkpoint = 250000
    num_checkpoints = 40

    non_normalized_min_transition_value = 0.2
    min_action_prob = 0.1


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


    simulation.run_estimation_error(
        num_experiments=num_experiments,
        num_samples_to_discard=100,
        num_samples_checkpoint=num_samples_checkpoint,
        num_checkpoints=num_checkpoints,
        min_action_prob=min_action_prob
        )

    print("Ciao")

