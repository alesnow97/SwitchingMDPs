import numpy as np
import os
import json
import itertools

from switchingBanditEnv import SwitchingBanditEnv


def compute_min_svd(reference_matrix):
    _, s, _ = np.linalg.svd(reference_matrix, full_matrices=True)
    return min(s)


def compute_second_eigenvalue(transition_matrix):
    evals, _ = np.linalg.eig(transition_matrix)
    evals = evals.real
    evals[np.isclose(evals, 1)] = -1
    return max(evals)


def save_files(switching: SwitchingBanditEnv, store_path):
    state_action_reward_matrix = switching.state_action_reward_matrix.reshape(
        -1)
    np.savetxt(f'{store_path}/transition_matrix.txt', switching.transition_matrix)
    np.savetxt(f'{store_path}/reference_matrix.txt', switching.reference_matrix)
    np.savetxt(f'{store_path}/state_action_reward_matrix.txt', state_action_reward_matrix)


def load_files(bandit_to_load_path, bandit_num, estimation_error_exp):

    if os.path.exists(bandit_to_load_path):

        f = open(bandit_to_load_path + f'/bandit{bandit_num}/bandit_info.json')
        data = json.load(f)
        print(data)

        if estimation_error_exp:
            return data['num_states'], data['num_actions'], data['num_obs'], \
                np.array(data['transition_matrix']), \
                np.array(data['reference_matrix']), \
                np.array(data['state_action_reward_matrix']), None
        else:
            return data['num_states'], data['num_actions'], data['num_obs'],\
                np.array(data['transition_matrix']), \
                np.array(data['reference_matrix']), \
                np.array(data['state_action_reward_matrix']), \
                np.array(data['possible_rewards'])


def find_optimal_exploration_length(switching: SwitchingBanditEnv, horizon_length, delta):
    rho_min = switching.stationary_distribution.min()
    epsilon = switching.transition_matrix.min()
    sigma_min = switching.sigma_min
    second_eingenvalue = switching.second_eigenvalue
    num_states = switching.num_states
    num_actions = switching.num_actions
    num_obs = switching.num_obs
    L = 4 * num_states * (1 - epsilon)**2 / epsilon**3 + np.sqrt(num_states)

    sqrt_numerator = np.sqrt(num_states * np.log(num_actions**2 * num_obs**2 / delta))
    sqrt_denominator = np.sqrt(1 - second_eingenvalue**(num_actions**2))

    # exploration_horizon = horizon_length*L*num_actions**2*num_obs * sqrt_numerator / (sigma_min * rho_min * sqrt_denominator)
    # multiplier = num_actions**2*num_obs * sqrt_numerator / (sigma_min * rho_min * sqrt_denominator)
    multiplier = num_actions*num_obs * sqrt_numerator / (sigma_min * sqrt_denominator)
    multiplier = num_actions*num_obs * sqrt_numerator / sqrt_denominator
    exploration_horizon = horizon_length * multiplier / 8

    # print(f"The value of L is {L}")
    print(f"The value of rho_min is {rho_min}")
    print(f"The value of multiplier is {multiplier}")
    # print(f"The value of epsilon is {epsilon}")
    print(f"The value of sigma_min is {sigma_min}")
    print(f"The value of sqrt_numerator is {sqrt_numerator}")
    print(f"The value of sqrt_denominator is {sqrt_denominator}")

    return exploration_horizon ** (2 / 3)


def find_best_permutation(real_matrix: np.ndarray, estimated_matrix: np.ndarray):
    # Define your data and error metric
    data = [i for i in range(real_matrix.shape[1])]
    error_metric = lambda x: np.sum(np.absolute(real_matrix.reshape(-1) - x.reshape(-1)))

    # Generate all permutations of the data
    permutations = itertools.permutations(data)

    # Find the permutation with the lowest error
    min_error = float('inf')
    min_permutation = None
    for permutation in permutations:
        current_perm_matrix = estimated_matrix[:, permutation]
        error = error_metric(current_perm_matrix)
        if error < min_error:
            min_error = error
            min_permutation = permutation

    print("Permutation with lowest error:", min_permutation)
    print("Lowest error:", min_error)
    return min_permutation, min_error

