import math
import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import t
import json

plt.style.use('seaborn-bright')

# visualization library
sns.set_context(rc={"font.family": 'sans',
                    "font.size": 12,
                    "axes.titlesize": 25,
                    "axes.labelsize": 24,
                    "ytick.labelsize": 15,
                    "xtick.labelsize": 15,
                    "lines.linewidth": 3,
                    })

def ci2(mean, std, n, conf=0.025):
    # Calculate the t-value
    t_value = t.ppf(1 - conf, n - 1)
    print(f"T value now is {t_value} with conf {conf}")

    # Calculate the margin of error
    margin_error = t_value * std / math.sqrt(n)

    # Calculate the lower and upper bounds of the confidence interval
    lower_bound = mean - margin_error
    upper_bound = mean + margin_error
    return lower_bound, upper_bound

if __name__ == '__main__':
    paths_to_read_from = ['experiments_estimation_error_all_arms/5states_10actions_10obs/bandit1/exp1',
                          'experiments_estimation_error_all_arms/10states_10actions_10obs/bandit0/exp1',
                          'experiments_estimation_error_all_arms/15states_10actions_10obs/bandit0/exp2',
                          'experiments_estimation_error_v2/5states_8actions_10obs/bandit0/exp1']

    # fig, axs = plt.plot(figsize=(20, 6))  # , sharex=True, sharey=True)
    fig, axs = plt.subplots(1, 2,
                            figsize=(20, 6))  # , sharex=True, sharey=True)
    x_axis = None
    exp_data = []
    num_experiments = 10

    # f = open(paths_to_read_from + '/exp_info.json')
    # data = json.load(f)
    # num_experiments = data['num_experiments']
    line_plot_colors1 = ['b', 'g', 'r']
    starting_index = 1

    for j, path in enumerate(paths_to_read_from):
        if j != 3:
            # Opening JSON file
            f = open(path + '/exp_info.json')
            # returns JSON object as
            # a dictionary
            data = json.load(f)
            # num_selected_arms = data['num_selected_arms']
            estimation_errors = np.array(data['estimation_errors'])
            num_states = data['num_states']
            if x_axis is None:
                x_axis = estimation_errors[0, starting_index:, 0]

            # estimation_errors = estimation_errors[:, 10:, 1].reshape(num_experiments, -1)
            estimation_errors = (estimation_errors[:, starting_index:, 1] / num_states).reshape(num_experiments, -1)

            mean_estimation_errors = np.mean(estimation_errors, axis=0)
            std_estimation_errors = np.std(estimation_errors, axis=0, ddof=1)
            lower_bound, upper_bound = ci2(mean_estimation_errors, std_estimation_errors, num_experiments)
            # lower_bound, upper_bound = mean_estimation_errors - std_estimation_errors, mean_estimation_errors + std_estimation_errors
            axs[0].plot(x_axis, mean_estimation_errors, line_plot_colors1[j], label=f'{num_states} states')
            axs[0].fill_between(x_axis, lower_bound, upper_bound, color=line_plot_colors1[j], alpha=0.2)
            axs[0].legend(prop={'size': 18})

    axs[0].set_title('(a)')
    #axs[i].legend()
    #axs[i].xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x/10000}'))
    #axs[i].xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    #axs[i].xaxis.set_minor_formatter(ticker.ScalarFormatter(useMathText=True))
    #axs[i].spines['left'].set_linewidth(2)
    axs[0].spines['left'].set_linewidth(2)
    axs[0].spines['bottom'].set_linewidth(2)
    axs[0].spines['top'].set_linewidth(1)
    axs[0].spines['right'].set_linewidth(1)

    axs[0].spines['left'].set_capstyle('butt')
    axs[0].spines['bottom'].set_capstyle('butt')
    axs[0].spines['top'].set_capstyle('butt')
    axs[0].spines['right'].set_capstyle('butt')
    # axs[0].ticklabel_format(useOffset=True)
    axs[0].set_xlabel('t')
    axs[0].set_ylabel('Estimation Error')
    axs[0].grid()
        # x_axis = None


    #paths_to_read_from = 'experiments_estimation_error_v2/5states_8actions_10obs/bandit0/exp1'

    # fig, axs = plt.plot(figsize=(20, 6))  # , sharex=True, sharey=True)
    x_axis = None
    exp_data = []

    f = open(paths_to_read_from[3] + '/exp_info.json')
    exp_info = json.load(f)
    num_experiments = exp_info['num_experiments']
    num_states = 5
    # line_plot_colors = []
    j = 0
    labels = ['low $\sigma_{\min}$', 'high $\sigma_{\min}$']
    line_plot_colors2 = ['g', 'r']
    print(exp_info)
    for i, path in enumerate(os.listdir(paths_to_read_from[3])):

        if path.endswith('arm.json'):
            # Opening JSON file
            f = open(paths_to_read_from[3] + '/' + path)
            # returns JSON object as
            # a dictionary
            data = json.load(f)
            estimation_errors = np.array(data)
            if x_axis is None:
                x_axis = estimation_errors[0, :, 0]
            #estimation_errors = estimation_errors[:, :, 1].reshape(num_experiments, -1)
            estimation_errors = (estimation_errors[:, :, 1] / num_states).reshape(num_experiments, -1)

            mean_estimation_errors = np.mean(estimation_errors, axis=0)
            std_estimation_errors = np.std(estimation_errors, axis=0, ddof=1)
            lower_bound, upper_bound = ci2(mean_estimation_errors, std_estimation_errors, num_experiments)
            axs[1].plot(x_axis, mean_estimation_errors, line_plot_colors2[j], label=labels[j])
            axs[1].fill_between(x_axis, lower_bound, upper_bound, color=line_plot_colors2[j], alpha=0.2)
            j += 1
    axs[1].legend(prop={'size': 18})
    axs[1].set_title('(b)')
    #axs[i].legend()
    #axs[i].xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x/10000}'))
    #axs[i].xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    #axs[i].xaxis.set_minor_formatter(ticker.ScalarFormatter(useMathText=True))
    #axs[i].spines['left'].set_linewidth(2)
    axs[1].spines['left'].set_linewidth(2)
    axs[1].spines['bottom'].set_linewidth(2)
    axs[1].spines['top'].set_linewidth(1)
    axs[1].spines['right'].set_linewidth(1)

    axs[1].spines['left'].set_capstyle('butt')
    axs[1].spines['bottom'].set_capstyle('butt')
    axs[1].spines['top'].set_capstyle('butt')
    axs[1].spines['right'].set_capstyle('butt')
    # axs[1].ticklabel_format(useOffset=True)
    axs[1].set_xlabel('t')
    axs[1].set_ylabel('Estimation Error')
    axs[1].grid()

    plt.tight_layout()
    plt.show()

    #axs[i].set_title(f"{i}-th exp")