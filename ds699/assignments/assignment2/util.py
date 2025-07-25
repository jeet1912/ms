import matplotlib.pyplot as plt
import numpy as np


def plot_return(return_list, save_plot_name, total_reward):
    """
    plot and save the image
    :param save_plot_name: name of the saved plot
    :param total_reward: total reward for each episode
    """
    episodes_list = list(range(len(return_list)))
    #safe_returns = [min(max(r, -1e6), 1e6) for r in return_list] #for overflow errors when n_steps > 1
    plt.plot(episodes_list, return_list)

    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    # use log scale for y axis, symlog means symmetric log scale
    plt.yscale('symlog')

    plt.title(f'reward:{total_reward}')
    plt.savefig(save_plot_name + '.png')


def actions_of_tabular_q(tabular_q):
    """
    print the actions of the tabular_q, 12 actions per line
    :param tabular_q: q table
    :return:
    """
    actions = []
    for i, state_actions in enumerate(tabular_q):
        actions.append(np.argmax(state_actions))
    print('actions:')
    for j in range(4):
        print(actions[j * 12:(j + 1) * 12])

