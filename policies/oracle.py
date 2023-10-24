import numpy as np

from switchingBanditEnv import SwitchingBanditEnv
from switchingBanditEnv_aistats import SwitchingBanditEnvAistats


class OraclePolicy:

    def __init__(self, switching_env: SwitchingBanditEnvAistats):

        self.transition_matrix = switching_env.transition_matrix
        self.state_action_reward_matrix = switching_env.state_action_reward_matrix
        self.possible_rewards = switching_env.possible_rewards
        self.belief = np.array([1/switching_env.num_states] * switching_env.num_states)
        print(self.belief)

    def choose_arm(self):
        current_state_arm_reward = self.state_action_reward_matrix
        # print(f"This should be {self.switching_env.num_states*self.switching_env.num_actions} "
        #       f"and it is {self.switching_env.state_action_reward_matrix.sum()}")
        scaled_matrix = self.belief[:, None, None] * current_state_arm_reward
        # print(scaled_matrix.sum())
        # print(f"Now this should be the one of before divided by {self.switching_env.num_states}")
        scaled_matrix = scaled_matrix * self.possible_rewards[None, None, :]
        #print(scaled_matrix)
        reduced_scaled_matrix = scaled_matrix.sum(axis=2)
        final_vector = reduced_scaled_matrix.sum(axis=0)
        chosen_action = np.argmax(final_vector)
        #print(reduced_scaled_matrix)
        #print(reduced_scaled_matrix.sum(axis=0))
        return chosen_action

    def update(self, pulled_arm, observerd_reward_index):
        observation_distribution = self.state_action_reward_matrix[:, pulled_arm, observerd_reward_index].reshape(-1)

        scaled_belief = self.belief * observation_distribution
        transitioned_belief = scaled_belief @ self.transition_matrix.T

        self.belief = transitioned_belief / transitioned_belief.sum()

