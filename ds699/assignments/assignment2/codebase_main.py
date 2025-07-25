import json
import statistics

import warnings
warnings.filterwarnings("ignore")

import gymnasium as gym
import pygame
import numpy as np
import random
from get_args import get_args
from codebase_Sarsa import sarsa
from codebase_Q_learning import q_learning
from codebase_train import render_single, evaluate_policy
from codebase_cliff_walking_test import test_each_action, test_moves


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
    :param args: parameters claimed in get_args.py file
    :return:
    """
    # initialize the environment
    env = gym.make("CliffWalking-v0")

    # total_rewards: initialize total reward list for plot
    total_rewards = []

    for seed in range(args.seeds):
        # set random seeds
        np.random.seed(seed)
        random.seed(seed)

        if args.method == 'test-cliff-walking':

            # test the environment with each action
            #test_each_action(env)

            ############################
            # Your Code #
            # Create a list of actions to direct the environment moving from the initial state to the most right-top corner state
            # Call the test_moves function to test if your series of actions are correct
            
            # 1.b
            #actions = [0,0,1,0,1,1,1,1,1,1,1,1,1,1,1,1]
            # 1.c
            actions = [0,1,1,1,1,1,1,1,1,1,1,1,2]
            test_moves(env,actions)

            ############################



            ############################
            # Your Code #
            # Your optimal moves: move from the initial state to the goal state with the maximum reward
            # Call the test_moves function to test if your series of actions is optimal



            ############################
            exit(0)

        elif args.method == 'sarsa':
            # call the sarsa function to get the q table
            tabular_q = sarsa(env, num_episode=args.num_episode, gamma=args.gamma,
                              alpha=args.alpha, init_epsilon=args.init_epsilon,
                              num_steps=args.num_steps, init_q_value=args.init_q_value)
        elif args.method == 'q-learning':
            # call the q learning function to get the q table
            tabular_q = q_learning(env, num_episode=args.num_episode, gamma=args.gamma, alpha=args.alpha,
                                   init_epsilon=args.init_epsilon)
        else:
            raise ValueError('Unknown method')

        # uncomment the following line to print the q table
        print('q table: \n', tabular_q)

        # uncomment the following line to render the environment
        # total_reward = render_single(env, tabular_q, 100)

        # uncomment the following line to just evaluate the policy without rendering
        total_reward = evaluate_policy(env, tabular_q, 100)

        total_rewards.append(total_reward)

    # uncomment the following line to print the mean and std of the total rewards
    #print('mean and std:', statistic_epsilon_analysis(total_rewards))


if __name__ == "__main__":

    # read in arguments
    args = get_args()
    main(args)
