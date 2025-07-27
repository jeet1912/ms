import torch

from codebase_torch_practice import test_torch_NN

import json
import statistics

import gymnasium as gym
import numpy as np
import random
from get_args import get_args
from codebase_CartPole_test import test_each_action, test_moves, init_states
from codebase_DQN import test_dqn

def statistic_epsilon_analysis(collect_rewards):
    """
    calculate the mean and std of the total rewards for multiple seeds
    :param collect_rewards: the total rewards for multiple seeds
    :return: mean and standard deviation of the total rewards
    """
    filtered_rewards = [r for r in collect_rewards if r > -100]
    mean_r = statistics.mean(filtered_rewards)
    std_r = statistics.stdev(filtered_rewards) if len(filtered_rewards) > 1 else 0
    return mean_r, std_r


def main(args):
    """
    main function to run sarsa and q learning
    :param args:
    :return:
    """

    for seed in range(args.seeds):
        # set random seeds
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

        if args.method == 'test-PyTorch':
            test_torch_NN(args)
            return
        elif args.method == 'test-CartPole':
            env = gym.make("CartPole-v1")

            # print the initial states
            # comment other 2 parts if you only want to run one of the following part
            #init_states(env)

            # test the environment with each action
            # comment other 2 parts if you only want to run one of the following part
            #test_each_action(env)

            # test the environment with different policies.
            # comment other 2 parts if you only want to run one of the following part
            #test_moves(env, 'random')
            #test_moves(env, 'right')
            test_moves(env, 'left')

            return

        elif args.method == 'DQN':

            # initialize the environment
            env = gym.make("CartPole-v1")
            # get the state dimension and number of actions
            state_dim = env.observation_space.shape[0]
            n_action = env.action_space.n
            # test the DQN algorithm
            test_dqn(args, env, state_dim, n_action)
        else:
            raise ValueError('Unknown method')


if __name__ == "__main__":
    # read in arguments

    args = get_args()
    main(args)

# test_torch_NN()
