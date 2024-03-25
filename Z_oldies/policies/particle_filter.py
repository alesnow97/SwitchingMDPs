import numpy as np
from Z_oldies.switchingBanditEnv_aistats import SwitchingBanditEnvAistats


class ParticleFilter:

    def __init__(self, switching_env: SwitchingBanditEnvAistats, total_horizon,
                 num_particles, lowest_prob, num_lowest_prob, dirichlet_prior):
        self.real_transition_matrix = switching_env.transition_matrix
        self.real_transition_stationary_distribution = switching_env.transition_stationary_distribution
        self.num_states = switching_env.num_states
        self.num_actions = switching_env.num_actions
        self.num_obs = switching_env.num_obs
        self.possible_rewards = switching_env.possible_rewards
        self.total_horizon = total_horizon

        self.state_action_reward_matrix = switching_env.state_action_reward_matrix
        self.reward_observation_matrix = self.state_action_reward_matrix * self.possible_rewards[None, None, :]

        self.num_particles = num_particles
        self.lowest_prob = lowest_prob
        self.num_lowest_prob = num_lowest_prob   # self.num_particles * something
        self.dirichlet_prior = dirichlet_prior
        self.dirichlets = None
        self.weights = None
        self.old_states = None
        self.current_states = None
        self.new_states = None

        self.reset()

    def reset(self):
        self.action_reward_list = []
        self.t = 0
        self.my_count = 0
        self.weights = np.array([1 / self.num_particles for _ in range(self.num_particles)])

        self.old_states = np.ones(self.num_particles, dtype="int64")
        self.current_states = np.random.randint(low=0, high=self.num_states, size=self.num_particles)
        self.new_states = np.ones(self.num_particles, dtype="int64")

        if self.dirichlet_prior is not None:
            dirichlet_prior = self.dirichlet_prior
        else:
            dirichlet_prior = np.ones((self.num_states, self.num_states))

        self.dirichlets = [dirichlet_prior for _ in range(self.num_particles)]
        self.count_pulled_arms = np.zeros(shape=self.num_actions)

    def choose_arm(self):
        action_obs_tensor = np.zeros((self.num_actions, self.num_obs))
        for particle in range(self.num_particles):
            action_obs_tensor += self.weights[particle] * self.reward_observation_matrix[
                self.current_states[particle]]
        chosen_action = np.argmax(action_obs_tensor.sum(axis=1))
        self.count_pulled_arms[chosen_action] += 1
        return chosen_action

    def update(self, pulled_arm, observed_reward_index):
        for particle in range(self.num_particles):
            old_particle_state = self.old_states[particle]
            particle_state = self.current_states[particle]

            particle_dirichlet = self.dirichlets[particle]

            old_state_dirichlet = particle_dirichlet[old_particle_state]
            if (old_state_dirichlet == 0).sum() > 0:
                indices = np.where(old_state_dirichlet == 0)
                # Substitute the old value with the new value at the found indices
                old_state_dirichlet[indices] = 1
            transition_vect_old_state = np.random.dirichlet(
                old_state_dirichlet)

            current_state_dirichlet = particle_dirichlet[particle_state]
            if (current_state_dirichlet == 0).sum() > 0:
                indices = np.where(current_state_dirichlet == 0)
                # Substitute the old value with the new value at the found indices
                current_state_dirichlet[indices] = 1
            # print(f"Current state dirichlet is {current_state_dirichlet}")
            transition_vect = np.random.dirichlet(current_state_dirichlet)

            new_particle_state = \
                np.random.multinomial(n=1, pvals=transition_vect, size=1)[
                    0].argmax()
            self.new_states[particle] = new_particle_state
            self.dirichlets[particle][particle_state, new_particle_state] += 1

            # update weight
            # print(f"Observation matrix is
            # {observation_matrix[particle_state,
            # chosen_action, observation]}")

            probabilities_over_states = self.state_action_reward_matrix[:, pulled_arm,
                                        observed_reward_index]

            if self.t == 0:
                numerator = self.state_action_reward_matrix[
                                particle_state, pulled_arm, observed_reward_index] * (
                                    1 / self.num_states)
                denominator = probabilities_over_states * np.array(
                    [1 / self.num_states for _ in range(self.num_states)])
                denominator = denominator / denominator.sum()
                denominator = denominator[particle_state]
            else:
                numerator = self.state_action_reward_matrix[
                                particle_state, pulled_arm, observed_reward_index] * \
                            transition_vect_old_state[particle_state]
                denominator = probabilities_over_states * transition_vect_old_state
                denominator = denominator / denominator.sum()
                denominator = denominator[particle_state]

            self.weights[particle] = self.weights[particle] * numerator / denominator

        self.weights = self.weights / self.weights.sum()

        if self.t % 500 == 0:
            print(f"t is {self.t}, {self.my_count}")
            print(self.count_pulled_arms)
            if self.t % 2000 == 0:
                self.compute_errors()

        self.old_states = self.current_states
        self.current_states = self.new_states

        # print(f"Min weight is {np.min(weights)}")
        ess = 1 / (self.weights ** 2).sum()

        if ess < 20:
            self.my_count += 1
            self.update_particles()
            ess = 1 / (self.weights ** 2).sum()
            # print(f"The effective Sample Size after update is {ess}")

        # if np.count_nonzero(self.weights < self.lowest_prob) > self.num_lowest_prob:
        #     ess = 1 / (self.weights ** 2).sum()
        #     print(f"The effective Sample Size before update is {ess}")
        #     self.update_particles()
        #     ess = 1 / (self.weights ** 2).sum()
        #     print(f"The effective Sample Size after update is {ess}")

        self.t += 1

    def update_particles(self):
        sampled_particles = \
            np.random.multinomial(n=self.num_particles, pvals=self.weights, size=1)[0]
        new_dirichlets = []
        new_weights = []
        new_current_states = []
        new_old_states = []
        for i, sample in enumerate(sampled_particles):
            for j in range(sample):
                new_dirichlets.append(self.dirichlets[i])
                new_weights.append(self.weights[i])
                new_current_states.append(self.current_states[i])
                new_old_states.append(self.old_states[i])
        self.dirichlets = new_dirichlets
        weights = np.array(new_weights)
        self.weights = weights / weights.sum()
        self.old_states = np.array(self.old_states)
        self.current_states = np.array(new_current_states)
        # print(f"Min weight after resampling is {np.min(weights)}")

    def compute_errors(self):
        sorted_indices = np.argsort(self.weights)
        sorted_weights = self.weights[sorted_indices]
        print(f"Starting")
        for j in range(self.num_particles - 2, self.num_particles):
            index = sorted_indices[j]
            current_dirichlet = self.dirichlets[index]
            current_dirichlet = current_dirichlet / current_dirichlet.sum(
                axis=1)[:, None]
            flatten_dirichlet = current_dirichlet.reshape(-1)
            distance_matrix = np.absolute(
                flatten_dirichlet - self.real_transition_matrix.reshape(-1))
            matrix_err_lstsq = np.sum(distance_matrix)
            print(f"Current Dirichlet is {current_dirichlet}")
            print(f"real transition matrix is {self.real_transition_matrix}")
            print(f"Distance vector with lstsq is {matrix_err_lstsq}")

