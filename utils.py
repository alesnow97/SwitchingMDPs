import json
import os

import numpy as np
from itertools import product, combinations_with_replacement

from pomdp_env.POMDP import POMDP


def load_pomdp(pomdp_to_load_path, pomdp_num):

    if os.path.exists(pomdp_to_load_path):

        f = open(pomdp_to_load_path + f'/pomdp{pomdp_num}/pomdp_info.json')
        data = json.load(f)
        f.close()
        # print(data)

        loaded_pomdp = POMDP(num_states=data['num_states'],
                             num_actions=data['num_actions'],
                             num_observations=data['num_obs'],
                             state_action_transition_matrix=np.array(data["state_action_transition_matrix"]),
                             state_action_observation_matrix=np.array(data["state_action_observation_matrix"]),
                             real_min_transition_value=data["real_min_transition_value"],
                             non_normalized_min_transition_value=data["non_normalized_min_transition_value"],
                             possible_rewards=np.array(data["possible_rewards"]),
                             transition_multiplier=data["transition_multiplier"],
                             observation_multiplier=data["observation_multiplier"]
                             )

        return loaded_pomdp


def load_pomdp_basic_info(pomdp_to_load_path, pomdp_num,
                          state_discretization_step,
                          action_discretization_step,
                          min_action_prob):

    base_path = pomdp_to_load_path + f"/pomdp{pomdp_num}/regret"
    basic_info_path = f"/{state_discretization_step}stst_{action_discretization_step}acst_{min_action_prob}_minac/basic_info.json"
    file_to_open_path = base_path + basic_info_path

    if os.path.exists(pomdp_to_load_path):

        f = open(file_to_open_path)
        data = json.load(f)
        # print(data)

        return (np.array(data["discretized_belief_states"]),
                np.array(data["discretized_action_space"]),
                np.array(data["real_belief_action_belief"]),
                np.array(data["real_optimal_belief_action_mapping"]),
                np.array(data["initial_discretized_belief"]),
                data["initial_discretized_belief_index"])

# def compute_min_svd(reference_matrix):
#     _, s, _ = np.linalg.svd(reference_matrix, full_matrices=True)
#     print(f"Dimension of s is {len(s)}")
#     return min(s)


def compute_second_eigenvalue(transition_matrix):
    evals, _ = np.linalg.eig(transition_matrix)
    evals = evals.real
    evals[np.isclose(evals, 1)] = -1
    return max(evals)


def dispositions_with_repetitions(array_size, possible_values):
    # Generate all possible combinations with replacement
    combinations = product(range(possible_values), repeat=array_size)
    # Convert combinations to a NumPy array
    result = np.array(list(combinations), dtype=np.float16)
    return result


def discretize_continuous_space(array_size, epsilon, min_value=1.0):

    num_bins = int((1 / epsilon))
    all_combinations = dispositions_with_repetitions(array_size=array_size, possible_values=num_bins + 1)

    discretized_beliefs = None
    for combination in all_combinations:
        #digit_array = int_to_array(combination, num_states, num_bins)
        if combination.sum() == num_bins:
            if discretized_beliefs is None:
                discretized_beliefs = combination.reshape(1, -1)
            else:
                discretized_beliefs = np.vstack([discretized_beliefs, combination])

    discretized_beliefs = discretized_beliefs / num_bins

    print(f"Data type is {discretized_beliefs.dtype}")


    if min_value != 1.0:
        filtered_array = discretized_beliefs[discretized_beliefs.min(axis=1) >= min_value]
        return filtered_array
    else:
        return discretized_beliefs


def find_closest_discretized_belief(discretized_beliefs, continuous_belief):

    belief_diff = discretized_beliefs - continuous_belief.reshape(1, -1)[:, None]
    belief_diff = belief_diff.reshape(-1, discretized_beliefs.shape[1])
    absolute_diff = np.absolute(belief_diff).sum(axis=1).reshape(-1)
    closest_belief_index = np.argmin(absolute_diff)
    closest_belief = discretized_beliefs[closest_belief_index]
    return closest_belief, closest_belief_index
