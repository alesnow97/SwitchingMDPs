import numpy as np
import json

from Z_SwitchingMDP.environment.MDPs import MDP


class SwitchingMDPs:

    def __init__(self,
                 num_modes,
                 num_states,
                 num_actions,
                 num_observations,
                 markov_chain_matrix=None,
                 state_action_transition_matrix_list=None,
                 state_action_observation_matrix_list=None,
                 possible_rewards=None,
                 modes_transition_multiplier=5,
                 transition_multiplier=5,
                 observation_multiplier=5
                 ):
        self.num_modes = num_modes
        self.modes_transition_multiplier = modes_transition_multiplier
        self.num_states = num_states
        self.num_actions = num_actions
        self.num_obs = num_observations
        self.state_action_transition_matrix_list = state_action_transition_matrix_list
        self.state_action_observation_matrix_list = state_action_observation_matrix_list
        self.possible_rewards = possible_rewards
        self.transition_multiplier = transition_multiplier
        self.observation_multiplier = observation_multiplier

        if self.possible_rewards is None:
            self.possible_rewards = np.random.permutation(
                np.linspace(start=0.0, stop=1.0, num=self.num_obs))
        if state_action_observation_matrix_list is not None:
            self.mdps = self.generate_mdps_from_params()
        else:
            self.mdps = self.generate_mdps()
            self.mode_state_action_transition_matrix = self.generate_mode_state_action_transition_matrix()
            self.mode_state_action_observation_matrix = self.generate_mode_state_action_observation_matrix()

        if markov_chain_matrix is not None:
            self.markov_chain_matrix = markov_chain_matrix
        else:
            self.markov_chain_matrix = self.generate_transition_matrix(transition_multiplier)

        self.mode_state_action_transition_matrix = self.generate_mode_state_action_transition_matrix()
        self.mode_state_action_observation_matrix = self.generate_mode_state_action_observation_matrix()


    def generate_mdps_from_params(self):
        mdps = []
        for i in range(len(self.state_action_observation_matrix_list)):
            current_mdp = MDP(num_states=self.num_states,
                              num_actions=self.num_actions,
                              num_observations=self.num_obs,
                              state_action_transition_matrix=self.state_action_transition_matrix_list[i],
                              state_action_observation_matrix=self.state_action_observation_matrix_list[i],
                              possible_rewards=self.possible_rewards)
            mdps.append(current_mdp)
        return mdps

    
    def generate_transition_matrix(self, transition_multiplier):
        # by setting specific design we give more probability to self-loops
        diag_matrix = np.eye(self.num_modes) * transition_multiplier
        for state in range(self.num_modes):
            outcome = np.random.random_integers(low=0, high=self.num_modes-1, size=1)[0]
            while outcome != state:
                outcome = \
                np.random.random_integers(low=0, high=self.num_modes-1, size=1)[
                    0]
            diag_matrix[state, outcome] += transition_multiplier * (2/3)
        markov_chain = np.random.random_integers(
            low=1, high=15, size=(self.num_modes, self.num_modes))
        markov_chain = markov_chain + diag_matrix
        transition_matrix = markov_chain / markov_chain.sum(axis=1)[:, None]
        return transition_matrix


    def generate_mdps(self):
        mdps = []
        for mode in range(self.num_modes):
            current_mdp = MDP(num_states=self.num_states,
                              num_actions=self.num_actions,
                              num_observations=self.num_obs,
                              state_action_transition_matrix=None,
                              state_action_observation_matrix=None,
                              possible_rewards=self.possible_rewards)
            mdps.append(current_mdp)
        return mdps

    def generate_mode_state_action_transition_matrix(self):
        mode_state_action_transition_matrix = np.empty(
            shape=(self.num_modes, self.num_states,
                   self.num_actions, self.num_states))

        for mode in range(self.num_modes):
            mode_state_action_transition_matrix[mode] = self.mdps[mode].state_action_transition_matrix

        return mode_state_action_transition_matrix

    def generate_mode_state_action_observation_matrix(self):
        mode_state_action_observation_matrix = np.empty(
            shape=(self.num_modes, self.num_states,
                   self.num_actions, self.num_obs))

        for mode in range(self.num_modes):
            mode_state_action_observation_matrix[mode] = self.mdps[
                mode].state_action_observation_matrix

        return mode_state_action_observation_matrix

    def get_next_state_reward(self, mode, state, action):
        current_mdp = self.mdps[mode]
        next_state = current_mdp.generate_next_state(
            state=state, action=action)
        current_reward = current_mdp.generate_current_reward(
            state=state, action=action)
        return next_state, current_reward

    def save(self):
        save_dict = {}
        save_dict["num_modes"] = self.num_modes
        save_dict["num_states"] = self.num_states
        save_dict["num_actions"] = self.num_actions
        save_dict["num_obs"] = self.num_obs

        save_dict["modes_transition_multiplier"] = self.modes_transition_multiplier
        save_dict["transition_multiplier"] = self.transition_multiplier
        save_dict["observation_multiplier"] = self.observation_multiplier

        save_dict["markov_chain_matrix"] = self.markov_chain_matrix.tolist()
        save_dict["mode_state_action_transition_matrix"] = self.mode_state_action_transition_matrix.tolist()
        save_dict["mode_state_action_observation_matrix"] = self.mode_state_action_observation_matrix.tolist()
        save_dict["possible_rewards"] = self.possible_rewards.tolist()

        # Convert and write JSON object to file
        with open("sample.json", "w") as outfile:
            json.dump(save_dict, outfile)












        
