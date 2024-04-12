import math
import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import ticker
from scipy.stats import t
import json

plt.style.use('fivethirtyeight')
sns.set_style(rc={"figure.facecolor":"white"})

def ci2(mean, std, n, conf=0.9):
    # Calculate the t-value
    t_value = t.ppf(1 - conf, n - 1)

    # Calculate the margin of error
    margin_error = t_value * std / math.sqrt(n)

    # Calculate the lower and upper bounds of the confidence interval
    lower_bound = mean - margin_error
    upper_bound = mean + margin_error
    return lower_bound, upper_bound


if __name__ == '__main__':
    # algorithms_to_use = ['sliding_w_UCB', 'epsilon_greedy', 'exp3S',
    #                      'our_policy']
    if os.getcwd().endswith("plots"):
        os.chdir("..")

    print(os.getcwd())

    base_path = ('ICML_experiments/3states_8actions_10obs/pomdp0/regret/'
                         '0.02stst_0.02_minac')

    basic_info_path = base_path + "/basic_info.json"
    oracle_opt_info_path = base_path + "/2500_init"
    num_experiments = 5
    num_episodes = 4

    # paths_to_read_from = ['ICML_experiments/4states_3actions_12obs/pomdp1/regret',
    #                       'experiments/5states_4actions_5obs/buono_5exp__lastexp/exp4']

    fig, axs = plt.subplots(1, 1, figsize=(20, 6))  # , sharex=True, sharey=True)
    # plot_titles = ['(a)', '(b)']

    oracle_collected_samples = None
    optimistic_collected_samples = None

    for experiment_num in range(num_experiments):
        current_exp_oracle_data = None
        current_exp_optimistic_data = None
        for episode_num in range(num_episodes):
            print(f"Experiment {experiment_num} and episode {episode_num}")

            # oracle
            f = open(oracle_opt_info_path + f'/oracle_{episode_num}Ep_{experiment_num}Exp.json')
            data = json.load(f)
            f.close()
            current_oracle_collected_samples = np.array(data["collected_samples"])

            # optimistic algorithm
            f = open(oracle_opt_info_path + f'/optimistic_{episode_num}Ep_{experiment_num}Exp.json')
            data = json.load(f)
            f.close()
            current_optimistic_collected_samples = np.array(data["collected_samples"])


            if current_exp_oracle_data is None:
                current_exp_oracle_data = current_oracle_collected_samples
            else:
                current_exp_oracle_data = np.vstack([current_exp_oracle_data, current_oracle_collected_samples])

            if current_exp_optimistic_data is None:
                current_exp_optimistic_data = current_optimistic_collected_samples
            else:
                current_exp_optimistic_data = np.vstack([current_exp_optimistic_data,
                                                      current_optimistic_collected_samples])

        if oracle_collected_samples is None:
            oracle_collected_samples = current_exp_oracle_data[:, 2].reshape(-1, 1)
        else:
            oracle_collected_samples = np.hstack(
                [oracle_collected_samples, current_exp_oracle_data[:, 2].reshape(-1, 1)])

        if optimistic_collected_samples is None:
            optimistic_collected_samples = current_exp_optimistic_data[:, 2].reshape(-1, 1)
        else:
            optimistic_collected_samples = np.hstack(
                [optimistic_collected_samples, current_exp_optimistic_data[:, 2].reshape(-1, 1)])


    optimistic_regret = oracle_collected_samples - optimistic_collected_samples
    optimistic_regret = optimistic_regret.T

    cumulative_optimistic_regret = np.cumsum(optimistic_regret, axis=1)
    mean_cumulated_optimistic_regret = np.mean(cumulative_optimistic_regret, axis=0)
    std_cumulative_optimistic_regret = np.std(cumulative_optimistic_regret, axis=0)
    lower_bound_opt, upper_bound_opt = ci2(mean_cumulated_optimistic_regret,
                                   std_cumulative_optimistic_regret, 2)

    x_axis = [i for i in range(optimistic_regret.shape[1])]

    axs.plot(mean_cumulated_optimistic_regret, 'c', label='OPT')
    axs.fill_between(x_axis,
                        lower_bound_opt,
                        upper_bound_opt,
                        color='c', alpha=.2)

    axs.legend()
    plt.tight_layout()
    plt.show()