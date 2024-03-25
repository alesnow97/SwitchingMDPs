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


dictionary = {"starting_checkpoint": 20000, "num_checkpoints": 51,
 "checkpoint_duration": 1000, "num_experiments": 1,
 "num_selected_arms": 3, "min_selected_arms": [2, 4, 7],
 "min_complexity_value": 13.036992612918764,
 "min_min_singular_value": 1.8900209117816145,
 "max_exploration_actions": [0, 1, 5],
 "max_complexity_value": 51.52145159182553,
 "max_min_singular_value": 0.4782510566738058}

if __name__ == '__main__':
    paths_to_read_from = 'experiments_estimation_error_v2/5states_8actions_10obs/bandit0/exp1'

    # fig, axs = plt.plot(figsize=(20, 6))  # , sharex=True, sharey=True)
    x_axis = None
    exp_data = []

    f = open(paths_to_read_from + '/exp_info.json')
    exp_info = json.load(f)
    num_experiments = exp_info['num_experiments']
    # line_plot_colors = []
    j = 0
    labels = ['low $\sigma$', 'high $\sigma$']
    print(exp_info)
    for i, path in enumerate(os.listdir(paths_to_read_from)):

        if path.endswith('arm.json'):
            # Opening JSON file
            f = open(paths_to_read_from + '/' + path)
            # returns JSON object as
            # a dictionary
            data = json.load(f)
            estimation_errors = np.array(data)
            if x_axis is None:
                x_axis = estimation_errors[0, :, 0]
            estimation_errors = estimation_errors[:, :, 1].reshape(num_experiments, -1)

            mean_estimation_errors = np.mean(estimation_errors, axis=0)
            std_estimation_errors = np.std(estimation_errors, axis=0, ddof=1)
            lower_bound, upper_bound = ci2(mean_estimation_errors, std_estimation_errors, num_experiments)
            plt.plot(x_axis, mean_estimation_errors, label=labels[j])
            plt.fill_between(x_axis, lower_bound, upper_bound, alpha=0.2)
            j += 1
    plt.legend()
    plt.tight_layout()
    plt.show()

    #axs[i].set_title(f"{i}-th exp")
