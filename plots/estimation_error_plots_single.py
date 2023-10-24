import math
import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import t
import json

plt.style.use('fivethirtyeight')

# visualization library
# sns.set(style="white", color_codes=True)
# sns.set_context(rc={"font.family": 'sans', "font.size": 24, "axes.titlesize": 24, "axes.labelsize": 24})


def ci(_mean, _std, n, conf=0.95):
  _adj_std = _std / np.sqrt(n)
  _low, _high = t.interval(conf, n-1, loc=_mean, scale=_adj_std)
  return _low, _high

def ci2(mean, std, n, conf=0.025):
    # Calculate the t-value
    t_value = t.ppf(1 - conf, n - 1)

    # Calculate the margin of error
    margin_error = t_value * std / math.sqrt(n)

    # Calculate the lower and upper bounds of the confidence interval
    lower_bound = mean - margin_error
    upper_bound = mean + margin_error
    return lower_bound, upper_bound

if __name__ == '__main__':
    paths_to_read_from = 'experiments_estimation_error/10states_10actions_10obs/bandit0/exp0'

    # fig, axs = plt.plot(figsize=(20, 6))  # , sharex=True, sharey=True)
    x_axis = None
    exp_data = []

    f = open(paths_to_read_from + '/exp_info.json')
    data = json.load(f)
    num_experiments = data['num_experiments']
    # line_plot_colors = []
    for i, path in enumerate(os.listdir(paths_to_read_from)):

        if path.endswith('arm.json') and not path.endswith('1_arm.json'):
            # Opening JSON file
            f = open(paths_to_read_from + '/' + path)
            # returns JSON object as
            # a dictionary
            data = json.load(f)
            num_selected_arms = data['num_selected_arms']
            estimation_errors = np.array(data['result'])
            if x_axis is None:
                x_axis = estimation_errors[0, :, 0]
            estimation_errors = estimation_errors[:, :, 1].reshape(num_experiments, -1)

            mean_estimation_errors = np.mean(estimation_errors, axis=0)
            std_estimation_errors = np.std(estimation_errors, axis=0, ddof=1)
            lower_bound, upper_bound = ci2(mean_estimation_errors, std_estimation_errors, num_experiments)
            plt.plot(x_axis, mean_estimation_errors, label=f'{num_selected_arms} arms')
            plt.fill_between(x_axis, lower_bound, upper_bound, alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.show()

    #axs[i].set_title(f"{i}-th exp")
