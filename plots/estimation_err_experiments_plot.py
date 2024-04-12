import math
import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import ticker
from scipy.stats import t
import json

import tikzplotlib

plt.style.use('seaborn-white')

#plt.style.use('fivethirtyeight')
#sns.set_style(rc={"figure.facecolor":"white"})

#styles = ['Solarize_Light2', '_classic_test_patch', '_mpl-gallery', '_mpl-gallery-nogrid', 'bmh', 'classic', 'dark_background', 'fast', 'fivethirtyeight', 'ggplot', 'grayscale', 'seaborn', 'seaborn-bright', 'seaborn-colorblind', 'seaborn-dark', 'seaborn-dark-palette', 'seaborn-darkgrid', 'seaborn-deep', 'seaborn-muted', 'seaborn-notebook', 'seaborn-paper', 'seaborn-pastel', 'seaborn-poster', 'seaborn-talk', 'seaborn-ticks', 'seaborn-white', 'seaborn-whitegrid', 'tableau-colorblind10']

# visualization library
#sns.set_style(style="bright", color_codes=True)
# sns.set_context(rc={"font.family": 'sans',
#                     "font.size": 12,
#                     "axes.titlesize": 25,
#                     "axes.labelsize": 24,
#                     "ytick.labelsize": 20,
#                     "xtick.labelsize": 20,
#                     "lines.linewidth": 4,
#                     })

def ci2(mean, std, n, conf=0.7):
    # Calculate the t-value
    t_value = t.ppf(1 - conf, n - 1)

    # Calculate the margin of error
    margin_error = t_value * std / math.sqrt(n)

    # Calculate the lower and upper bounds of the confidence interval
    lower_bound = mean - margin_error
    upper_bound = mean + margin_error
    return lower_bound, upper_bound

np.random.seed(200)


if __name__ == '__main__':
    # algorithms_to_use = ['sliding_w_UCB', 'epsilon_greedy', 'exp3S',
    #                      'our_policy']
    if os.getcwd().endswith("plots"):
        os.chdir("..")

    print(os.getcwd())

    base_path_first = ('ICML_experiments_error/3states_4actions_5obs/pomdp10/estimation_error')

    fig, axs = plt.subplots(1, 1, figsize=(20, 6))  # , sharex=True, sharey=True)
    # plot_titles = ['(a)', '(b)']

    # first exp
    num_checkpoints = 200
    checkpoint_size = 1000      # number of couples
    basic_info_path = base_path_first + f"/{checkpoint_size}_{num_checkpoints}cp_0.json"
    f = open(basic_info_path)
    data = json.load(f)
    f.close()
    frobenious_norm_first = np.array(data["error_frobenious_norm"])

    print(frobenious_norm_first.shape)

    mean_frobenious_first = frobenious_norm_first.mean(axis=0)
    std_frobenious_first = frobenious_norm_first.std(axis=0)
    lower_bound_first, upper_bound_first = ci2(mean_frobenious_first,
                                   std_frobenious_first, frobenious_norm_first.shape[0])

    x_axis = np.array([checkpoint_size*(i+1) for i in range(num_checkpoints)])
    x_axis_mask = np.array([i % 10 == 0 for i in range(num_checkpoints)])

    axs.plot(x_axis[x_axis_mask], mean_frobenious_first[x_axis_mask], 'c', label='3stat 4act 5obs')
    # axs.fill_between(x_axis,
    #                     lower_bound_first,
    #                     upper_bound_first,
    #                     color='c', alpha=.2)

    # second exp
    base_path_second = (
        'ICML_experiments_error/5states_3actions_8obs/pomdp4/estimation_error')
    num_checkpoints = 200
    checkpoint_size = 1000  # number of couples
    basic_info_path = base_path_second + f"/{checkpoint_size}_{num_checkpoints}cp_0.json"
    f = open(basic_info_path)
    data = json.load(f)
    f.close()
    frobenious_norm_second = np.array(data["error_frobenious_norm"])

    print(frobenious_norm_second.shape)

    mean_frobenious_second = frobenious_norm_second.mean(axis=0)
    std_frobenious_second = frobenious_norm_second.std(axis=0)
    lower_bound_second, upper_bound_second = ci2(mean_frobenious_second,
                                               std_frobenious_second,
                                               frobenious_norm_second.shape[0])

    x_axis = np.array(
        [checkpoint_size * (i + 1) for i in range(num_checkpoints)])
    x_axis_mask = np.array([i % 10 == 0 for i in range(num_checkpoints)])
    x_axis_mask[0] = False

    axs.plot(x_axis[x_axis_mask], mean_frobenious_second[x_axis_mask], 'r',
             label='5stat 3act 8obs')
    # axs.fill_between(x_axis,
    #                     lower_bound_second,
    #                     upper_bound_second,
    #                     color='c', alpha=.2)

    axs.legend()
    plt.tight_layout()
    plt.grid(True)

    show_figure = False
    if show_figure is True:
        plt.show()
    else:
        import tikzplotlib

        print(os.getcwd())
        tikzplotlib.save("estimation_err.tex")
    # or
    # tikzplotlib.save("estimation_err.tex", flavor="context")