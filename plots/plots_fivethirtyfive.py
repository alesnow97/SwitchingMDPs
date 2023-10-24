import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import ticker
from scipy.stats import t
import json

plt.style.use('fivethirtyeight')
sns.set_style(rc={"figure.facecolor":"white"})

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
    algorithms_to_use = ['sliding_w_UCB', 'epsilon_greedy', 'exp3S',
                         'our_policy']
    paths_to_read_from = ['experiments/3states_4actions_5obs/bandit11/buono_5exp',
                          'experiments/5states_4actions_5obs/buono_5exp__lastexp/exp4']

    fig, axs = plt.subplots(1, len(paths_to_read_from), figsize=(20, 6))  # , sharex=True, sharey=True)
    plot_titles = ['(a)', '(b)']
    exp_data = []
    for i, path in enumerate(paths_to_read_from):
        # Opening JSON file
        f = open(path + '/exp_info.json')
        # returns JSON object as
        # a dictionary
        data = json.load(f)
        oracle_list = np.array(data['rewards']['oracle'])
        sliding_w_UCB_list = np.array(data['rewards']['sliding_w_UCB'])
        epsilon_greedy_list = np.array(data['rewards']['epsilon_greedy'])
        exp3S_list = np.array(data['rewards']['exp3S'])
        our_policy_list = np.array(data['rewards']['our_policy'])

        oracle_rewards = oracle_list[:, :, 1]
        x_axis = [i for i in range(oracle_rewards.shape[1])]

        sliding_w_UCB_regret = np.mean(oracle_rewards - sliding_w_UCB_list[:, :, 1], axis=0)
        epsilon_greedy_regret = np.mean(
            oracle_rewards - epsilon_greedy_list[:, :, 1], axis=0)
        exp3S_regret = np.mean(oracle_rewards - exp3S_list[:, :, 1], axis=0)
        our_policy_regret = np.mean(oracle_rewards - our_policy_list[:, :, 1],
                                    axis=0)

        print(f"sliding_w_UCB regret {sliding_w_UCB_regret.sum()}")
        axs[i].plot(np.cumsum(sliding_w_UCB_regret), 'c', label='SW-UCB')

        print(f"Epsilon greedy regret {epsilon_greedy_regret.sum()}")
        axs[i].plot(np.cumsum(epsilon_greedy_regret), 'b', label='EPS-gr')

        print(f"Exp3S regret {exp3S_regret.sum()}")
        axs[i].plot(np.cumsum(exp3S_regret), 'g', label='Exp3.S')

        print(f"Our policy regret {our_policy_regret.sum()}")
        axs[i].plot(np.cumsum(our_policy_regret), 'r', label='Our')
        # for row in range(oracle_rewards.shape[0]):
        # axs[i].plot(np.cumsum(oracle_rewards - our_policy_list[:, :, 1]), 'r')
        # axs[i].fill_between(x_axis, np.cumsum(our_policy_low), np.cumsum(our_policy_high), alpha=0.2)

        axs[i].set_title(plot_titles[i])
        #axs[i].legend()
        #axs[i].xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x/10000}'))
        #axs[i].xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
        #axs[i].xaxis.set_minor_formatter(ticker.ScalarFormatter(useMathText=True))
        #axs[i].spines['left'].set_linewidth(2)
        axs[i].spines['left'].set_linewidth(2)
        axs[i].spines['left'].set_visible(True)
        axs[i].spines['bottom'].set_linewidth(2)
        axs[i].spines['top'].set_linewidth(1)
        axs[i].spines['right'].set_linewidth(1)

        axs[i].spines['left'].set_capstyle('butt')
        axs[i].spines['bottom'].set_capstyle('butt')
        axs[i].spines['top'].set_capstyle('butt')
        axs[i].spines['right'].set_capstyle('butt')
        # axs[i].ticklabel_format(useOffset=True)
        axs[i].set_xlabel('t')
        axs[i].set_ylabel('$\widehat{\mathcal{R}}(t)$')
        if i == 1:
            axs[i].legend()

    #lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    #lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    #fig.legend(lines, labels)

    #fig.legend(*axs[0].get_legend_handles_labels(),
    #           loc='upper center', ncol=4)

    plt.tight_layout()
    plt.show()
