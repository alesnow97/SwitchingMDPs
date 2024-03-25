import numpy as np

from pomdp_env.POMDP import POMDP
from pomdp_env.pomdp_simulations_new import POMDPSimulationNew
from utils import load_pomdp

if __name__ == '__main__':

    # run settings
    save_pomdp_info = False
    save_results = True
    to_load = True

    num_states = 3
    num_actions = 2
    num_observations = 5
    #horizon = 10000000
    num_experiments = 1
    num_samples_to_discard = 250
    num_samples_checkpoint = 20000
    num_checkpoints = 5

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

    simulation.run_estimation_error(num_experiments=num_experiments,
                   num_samples_to_discard=num_samples_to_discard,
                   num_samples_checkpoint=num_samples_checkpoint,
                   num_checkpoints=num_checkpoints)

    # check min_singular_value
    print("Ciao")


    #simulation = EstimationErrorSimulation(switching_mdps=switching_mdp)
    #simulation.run(horizon=horizon, num_experiments=1)





