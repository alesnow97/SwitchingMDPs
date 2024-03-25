import numpy as np

class StateBasedPolicy:

    def __init__(self,
                 num_states,
                 num_actions,
                 num_obs):
        self.num_states = num_states
        self.num_actions = num_actions
        self.num_obs = num_obs

        self.state_policy = self.generate_state_based_policy()


    def generate_state_based_policy(self):
        state_action_policy = np.random.random((self.num_states, self.num_actions))
        state_action_policy = state_action_policy / state_action_policy.sum(axis=1)[:, None]

        return state_action_policy


    def choose_arm(self, state):
        sampled_action = np.random.multinomial(n=1, pvals=self.state_policy[state], size=1)[0].argmax()
        return sampled_action

