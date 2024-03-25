import json
import os

import numpy as np

from pomdp_env.POMDP import POMDP


def load_pomdp(pomdp_to_load_path, pomdp_num):

    if os.path.exists(pomdp_to_load_path):

        f = open(pomdp_to_load_path + f'/pomdp{pomdp_num}/pomdp_info.json')
        data = json.load(f)
        # print(data)

        loaded_pomdp = POMDP(num_states=data['num_states'],
                             num_actions=data['num_actions'],
                             num_observations=data['num_obs'],
                             state_action_transition_matrix=np.array(data["state_action_transition_matrix"]),
                             state_action_observation_matrix=np.array(data["state_action_observation_matrix"]),
                             possible_rewards=np.array(data["possible_rewards"]),
                             transition_multiplier=data["transition_multiplier"],
                             observation_multiplier=data["observation_multiplier"]
                             )

        return loaded_pomdp

# def compute_min_svd(reference_matrix):
#     _, s, _ = np.linalg.svd(reference_matrix, full_matrices=True)
#     print(f"Dimension of s is {len(s)}")
#     return min(s)


def compute_second_eigenvalue(transition_matrix):
    evals, _ = np.linalg.eig(transition_matrix)
    evals = evals.real
    evals[np.isclose(evals, 1)] = -1
    return max(evals)