import matplotlib.pyplot as plt
import numpy as np


def plot_episode_rewards(episode_rewards, save_plot_name):
    """
    plot and save the image
    :param episode_rewards: total reward for each episode
    :param save_plot_name: name of the saved plot
    """
    # compute the average reward of 10 episodes
    avg_episode_rewards = [np.mean(episode_rewards[i:i + 10]) for i in range(len(episode_rewards) - 10)]

    plt.plot([i for i in range(len(episode_rewards))], episode_rewards)
    plt.plot([i for i in range(len(avg_episode_rewards))], avg_episode_rewards)
    plt.legend(['reward', 'avg_reward'])
    file_name = save_plot_name
    plt.savefig(file_name)
    print('finish')
    plt.close()
