import numpy as np

import utils

def compute_belief_action_belief_matrix(num_actions, num_obs,
                                        discretized_belief_states, len_discretized_beliefs,
                                        state_action_transition_matrix,
                                        state_action_observation_matrix):
    belief_action_belief_matrix = np.zeros(
        shape=(len_discretized_beliefs, num_actions, len_discretized_beliefs), dtype=np.float16)
    for i in range(len_discretized_beliefs):
        if i % 100 == 0:
            print(i)
        current_belief = discretized_belief_states[i]
        for action in range(num_actions):
            # just for debug purposes
            obs_probabilities = np.zeros(shape=(num_obs))
            for obs in range(num_obs):
                observation_distribution = state_action_observation_matrix[
                                           :,
                                           action,
                                           obs].reshape(-1)

                scaled_belief = current_belief * observation_distribution
                obs_probability = np.sum(scaled_belief)
                obs_probabilities[obs] = obs_probability
                # current_transition_matrix = self.pomdp.state_action_transition_matrix[
                #                             :, action, :]
                current_transition_matrix = state_action_transition_matrix[:, action, :]
                transitioned_belief = scaled_belief @ current_transition_matrix.T
                transitioned_belief = transitioned_belief / transitioned_belief.sum()

                discretized_transitioned_belief, discretized_transitioned_belief_index = (
                    utils.find_closest_discretized_belief(discretized_belief_states, transitioned_belief))

                belief_action_belief_matrix[i, action, discretized_transitioned_belief_index] += obs_probability

    return belief_action_belief_matrix


def compute_optimal_POMDP_policy(num_actions,
                                 discretized_belief_states,
                                 len_discretized_beliefs,
                                 ext_v_i_stopping_cond,
                                 min_action_prob,
                                 state_action_reward,
                                 belief_action_belief_matrix):

    u = np.zeros(shape=(len_discretized_beliefs))
    best_action_dist_per_u = np.zeros(shape=(len_discretized_beliefs, num_actions))

    du = np.zeros(shape=(len_discretized_beliefs))
    du[0], du[-1] = np.inf, -np.inf
    counter = 0
    while not du.max() - du.min() < ext_v_i_stopping_cond:
        if counter >= 100:
            raise ValueError("Convergence error")
        counter += 1
        new_u = np.zeros(shape=(len_discretized_beliefs))
        print(f"Len of discretized belief is {len_discretized_beliefs}")
        for j in range(len_discretized_beliefs):
            if j % 1000 == 0:
                print(j)
            current_belief = discretized_belief_states[j]

            current_u_a = np.zeros(shape=num_actions)
            for action in range(num_actions):
                state_reward = state_action_reward[:, action]
                mean_action_reward = np.multiply(current_belief, state_reward).sum()
                current_belief_action_probs = belief_action_belief_matrix[j, action, :]
                future_value_function_array = np.multiply(
                    current_belief_action_probs, u.reshape(1, -1))
                future_value_function = np.sum(future_value_function_array)
                current_u_a[action] = mean_action_reward + future_value_function

            best_action_dist_index = np.argmax(current_u_a)
            best_action_dist_prob = np.ones(shape=num_actions) * min_action_prob
            best_action_dist_prob[best_action_dist_index] = 1 - ((num_actions - 1) * min_action_prob)

            state_action_probs = np.outer(current_belief, best_action_dist_prob)
            mean_reward = np.multiply(state_action_reward, state_action_probs).sum()

            current_action_belief = belief_action_belief_matrix[j, :, :]
            current_belief_action_probs = np.multiply(current_action_belief, best_action_dist_prob.reshape(-1, 1))
            future_value_function_array = np.multiply(current_belief_action_probs, u.reshape(1, -1))
            future_value_function = np.sum(future_value_function_array)

            best_action_dist_per_u[j] = best_action_dist_prob
            new_u[j] = mean_reward + future_value_function

            # VERSION USING THE DISCRETIZED ACTION SPACE
            # current_u_a = np.zeros(shape=(len_discretized_action_space))
            # for z in range(len_discretized_action_space):
            #
            #     current_action_prob = discretized_action_space[z]
            #     state_action_probs = np.outer(current_belief, current_action_prob)
            #     mean_reward = np.multiply(state_action_reward, state_action_probs).sum()
            #
            #     current_action_belief = belief_action_belief_matrix[j, :, :]
            #     current_belief_action_probs = np.multiply(current_action_belief, current_action_prob.reshape(-1, 1))
            #     future_value_function_array = np.multiply(current_belief_action_probs, u.reshape(1, -1))
            #     future_value_function = np.sum(future_value_function_array)
            #
            #     current_u_a[z] = mean_reward + future_value_function
            #
            # best_action_dist_index = np.argmax(current_u_a)
            # best_action_dist_per_u[j] = discretized_action_space[best_action_dist_index]
            # new_u[j] = current_u_a[best_action_dist_index]

        du = new_u - u
        print(du.max() - du.min())
        u = new_u

    return best_action_dist_per_u


def inner_maximization(p_sa_hat, confidence_bound_p_sa, rank, min_transition_value):
    '''
    Find the best local transition p(.|s, a) within the plausible set of transitions as bounded by the confidence bound for some state action pair.
    Arg:
        p_sa_hat : (n_states)-shaped float array. MLE estimate for p(.|s, a).
        confidence_bound_p_sa : scalar. The confidence bound for p(.|s, a) in L1-norm.
        rank : (n_states)-shaped int array. The sorted list of states in descending order of value.
    Return:
        (n_states)-shaped float array. The optimistic transition p(.|s, a).
    '''
    # print('rank', rank)
    p_sa = np.array(p_sa_hat)
    max_value = 1 - min_transition_value * (p_sa.shape[0] - 1)
    p_sa[rank[0]] = min(max_value, p_sa_hat[rank[0]] + confidence_bound_p_sa / 2)
    rank_dup = list(rank)
    last = rank_dup.pop()
    # Reduce until it is a distribution (equal to one within numerical tolerance)
    while sum(p_sa) > 1 + 1e-9:
        # print('inner', last, p_sa)
        p_sa[last] = max(min_transition_value, 1 - sum(p_sa) + p_sa[last])
        # p_sa[last] = max(0, 1 - sum(p_sa) + p_sa[last])
        last = rank_dup.pop()
    # print('p_sa', p_sa)
    return p_sa



def compute_optimistic_MDP(
        num_states,
        num_actions,
        min_action_prob,
        state_action_transition_matrix,
        ext_v_i_stopping_cond,
        state_action_reward,
        confidence_bound,
        min_transition_value
):
    '''
    The extended value iteration which finds an optimistic MDP within the plausible set of MDPs and solves for its near-optimal policy.
    '''
    # Initial values (an optimal 0-step non-stationary policy's values)
    u = np.zeros(num_states)
    new_u = np.zeros(num_states)
    du = np.zeros(num_states)
    du[0], du[-1] = np.inf, -np.inf

    # Optimistic MDP and its epsilon-optimal policy
    p_tilde = np.zeros((num_states, num_actions, num_states))
    best_action_dist_per_state = np.zeros(shape=(num_states, num_actions))

    counter = 0
    while not du.max() - du.min() < ext_v_i_stopping_cond:
        counter += 1
        # Sort the states by their values in descending order
        rank = np.argsort(-u)
        for st in range(num_states):

            q_s_per_action = np.zeros(shape=num_actions)
            for ac in range(num_actions):
                # Optimistic transitions
                p_sa_tilde = inner_maximization(
                    state_action_transition_matrix[st, ac], confidence_bound,
                    rank, min_transition_value)
                q_sa = state_action_reward[st, ac] + (p_sa_tilde * u).sum()
                q_s_per_action[ac] = q_sa
                p_tilde[st, ac] = p_sa_tilde

            # best_action_dist_index = np.argmax(current_u_a)
            # best_action_dist_prob = np.ones(shape=num_actions) * min_action_prob
            # best_action_dist_prob[best_action_dist_index] = 1 - ((num_actions - 1) * min_action_prob)

            best_action_index = np.argmax(q_s_per_action)
            # q_s_action_dist = np.sum(np.multiply(discretized_action_space, q_s_per_action), axis=1)
            # q_s_action_dist = q_s_action_dist.reshape(-1)

            best_action_dist = np.ones(shape=num_actions) * min_action_prob
            best_action_dist[best_action_index] = 1 - ((num_actions - 1) * min_action_prob)

            # max_action_dist_index = np.argmax(q_s_action_dist)
            # best_action_dist_per_state[st] = discretized_action_space[max_action_dist_index]

            best_q_sa = np.sum(np.multiply(best_action_dist, q_s_per_action))
            new_u[st] = best_q_sa

        du = new_u - u
        u = new_u
        new_u = np.zeros(num_states)
        # print('u', state_value_hat, du.max() - du.min(), epsilon)
    return p_tilde, best_action_dist_per_state
