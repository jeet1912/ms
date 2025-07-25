import math

import numpy as np
from codebase_train import epsilon_greedy
from util import plot_return


def n_step(env, tabular_q, epsilon, action, gamma, steps, total_reward):
    """
    n-step sarsa
    :param env: environment
    :param tabular_q: q table
    :param epsilon: epsilon for epsilon greedy
    :param action: action to take for the first step
    :param gamma: gamma for decreasing the reward
    :param steps: number of steps for n-step sarsa
    :param total_reward: total reward for the episode.
    :return:
    next state: next state after n steps,
    action: next action to take for the next state,
    n_step_reward: n-step reward with gamma. It is used for updating the q table.
    done: terminal signal
    acted_steps: the number of steps took in the environment, different from the input steps if the episode terminates before steps
    total reward: cumulated total reward for the episode without gamma.
    """
    # acted_steps: number of steps taken in the environment
    acted_steps = 0
    # n_step_reward: n-step reward with gamma. It is used for updating the q table.
    n_step_reward = 0
    # done: terminal signal
    done = False


    ############################
    # Your Code #
    # Implement the n-step sample of sarsa, #
    # You need to stop the iteration if the episode terminates before n steps and record how many steps has been taken in the variable acted_steps.
    # The acted_steps is needed in the sarsa function for updating the q table. #
    # You need to call the env.step to get the next state, reward, and other information.
    # Please read the instruction carefully to get familiar with the env.step function. #
    # Please use while loop and call the epsilon_greedy function in codebase_train.py to finish this part. #

    ##########################
    # TODO: Implement the n-step sampling for SARSA. #
    # You will simulate up to `steps` steps starting from the current state, #
    # and accumulate the total discounted reward along the way.              #
    #
    # Instructions:
    # - Use a while-loop to iterate through the environment steps.
    # - At each step, use `env.step(action)` to interact with the environment.
    # - Use `epsilon_greedy(tabular_q, next_state, epsilon)` to select the next action.
    # - Track how many steps were actually taken in `acted_steps`.
    # - If the environment terminates (`done == True`), stop the loop early.
    # - For each step, accumulate the discounted reward into `n_step_reward` using the formula:
    #   γ^t * reward_t

    while acted_steps < steps and not done:
        next_state, reward, done, _, _ = env.step(action)
        total_reward += reward
        n_step_reward += (gamma**acted_steps) * reward
        acted_steps += 1
        if not done:
            action = epsilon_greedy(tabular_q,next_state,epsilon)



    ############################

    return next_state, action, n_step_reward, done, acted_steps, total_reward


def sarsa(env, num_episode, gamma, alpha, init_epsilon, num_steps, init_q_value=0):
    """
    Sarsa algorithm
    :param env:environment
    :param num_episode: number of episodes generate from the environment
    :param gamma: gamma for decreasing the reward
    :param alpha: alpha in sarsa algorithm to update q table
    :param init_epsilon: epsilon for epsilon greedy
    :param num_steps: number of steps for n-step sarsa
    :param init_q_value: initial q value for q table
    :return:
    tabular_q: q table for each state and action pair
    """

    # tabular_q: initialize q table
    # tabular_q[state][action] stores q value for state action pair
    print(init_q_value)
    tabular_q = np.ones(shape=(48, 4)) * init_q_value

    # total_rewards: initialize total reward list for plot
    total_rewards = []

    for episode in range(num_episode):

        # done: terminal signal, true for reaching terminal state
        done = False

        # total_reward: total reward for each episode
        total_reward = 0
        # episode_length: length of the episode
        episode_len = 0

        # state: initialize state from reset
        state, _ = env.reset()

        ############################
        # Your Code #
        # for the question epsilon effect. #
        # For the question in part II.(a), you do not need to do anything, just let epsilon = init_epsilon for all episode #
        # For the question in part II.(b), please modify the following line. You need to update epsilon using the number of episode according to the instruction #
        # II a
        #epsilon = init_epsilon
        ############################
        # II b
        epsilon = init_epsilon / (episode+1)

        # action: initialize an action from epsilon greedy policy and input
        action = epsilon_greedy(tabular_q, state, epsilon)


        ############################
        # Your Code #
        # Implement the sarsa algorithm
        # First, you need to call the n_step function to get the values for updating tarbar_q #
        # Then, update the q table tarbar_q with the n_step_reward, update the state, action, and episode_len #
        # Do not forget to update the total reward and episode_len #
        # Please use while loop to finish this part. #
        while not done: 
            next_state, next_action, n_step_reward, done, acted_steps, total_reward = n_step(env, tabular_q, epsilon, action, gamma, num_steps, total_reward)
            #print(f'state :{state}, action:{action}, reward:{total_reward}, done:{done}, next_state:{next_state}')
            tabular_q[state][action] += alpha*(n_step_reward + gamma*tabular_q[next_state][next_action] - tabular_q[state][action])
            episode_len += acted_steps
            total_reward = total_reward
            state = next_state
            action = next_action


        ############################

        # print the total reward of current episode
        print("Episode:", episode, "Total Reward: ", total_reward)
        # store total reward for each episode in the total_rewards list
        # tabular_q_analysis(tabular_q)
        total_rewards.append(total_reward)

    # plot the total_rewards in terms of the number of episodes, save the plot.
    # Modify the saved plot name if you need to save different names for different epsilon method #
    # Comment the following line if you do not want to save the plot. It can save the running time. #
    plot_return(total_rewards,
                save_plot_name=f'Sarsa steps_{num_steps}_alpha_{alpha}'
                               f'_init_q_value_{init_q_value}_init_epsilon_{init_epsilon}_epsilonChanges',
                total_reward=total_reward)
    return tabular_q
