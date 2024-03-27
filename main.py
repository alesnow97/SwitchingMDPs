import numpy as np

from pomdp_env.POMDP import POMDP
from pomdp_env.pomdp_simulations_new import POMDPSimulationNew
from utils import load_pomdp

if __name__ == '__main__':

    # run settings
    save_pomdp_info = False
    save_results = False
    to_load = False

    num_states = 3
    num_actions = 3
    num_observations = 5
    horizon = 100000
    num_experiments = 1

    # estimation error experiment
    num_samples_to_discard = 250
    num_samples_checkpoint = 20000
    num_checkpoints = 5

    # regret experiment
    ext_v_i_stopping_cond = 0.000001
    state_discretization_step = 0.1
    action_discretization_step = 0.05

    non_normalized_min_transition_value = 0.2
    min_action_prob = 0.05
    T_0 = 100000
    num_episodes = 10

    pomdp_to_load_path = f"ICML_experiments/{num_states}states_{num_actions}actions_{num_observations}obs/"
    pomdp_num = 0

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
                                    pomdp_num=0,
                                    save_pomdp_info=save_pomdp_info,
                                    save_results=save_results
                                    )

    simulation.run_regret_experiment(
        num_experiments=num_experiments,
        T_0=T_0,
        num_episodes=num_episodes,
        ext_v_i_stopping_cond=ext_v_i_stopping_cond,
        state_discretization_step=state_discretization_step,
        action_discretization_step=action_discretization_step,
        min_action_prob=min_action_prob
    )

    print("Ciao")

