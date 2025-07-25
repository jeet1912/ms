import time
import numpy as np
from util import actions_of_tabular_q

def epsilon_greedy(tabular_q, state, epsilon=0.0):
    """
    epsilon greedy policy to select action
    :param tabular_q: q table
    :param state: current state to select action
    :param epsilon: the probability to select a random action
    :return: action: the action to take according to the epsilon greedy policy
    """
    ############################
    # Your Code #
    # implement the epsilon greedy policy.
    # You can select random actions by using np.random.randint and limiting the random range.
    # Please use np.argmax from greedy selection

    # II a
    
    if np.random.uniform(0,1) < epsilon:
        action = np.random.randint(0,4)
    else:
        action = np.argmax(tabular_q[state])


    ############################
    return action


def greedy_policy(tabular_q, state):
    """
    Greedy policy to select action. It is used for rendering.
    :param tabular_q: q table
    :param state: current state to select action
    :return: action: the action to take according to the greedy policy
    """
    action = int(np.argmax(tabular_q[state]))

    return action


def render_single(env, tabular_q, max_steps=100):
    '''

    This function does not need to be modified.
    Renders policy once on environment. Watch your agent play!
    :param env: Environment to play on. Must have nS, nA, and P as attributes.
    :param tabular_q: q table
    :param max_steps: the maximum number of iterations
    :return: episode_reward: total reward for the episode
    '''

    episode_reward = 0
    actions_of_tabular_q(tabular_q)
    state, _ = env.reset()
    for t in range(max_steps):
        env.render()
        time.sleep(0.15)
        action = greedy_policy(tabular_q, state)
        state, reward, done, _, _ = env.step(action)
        episode_reward += reward
        if done:
            break
    env.render()
    if not done:
        print("The agent didn't reach a terminal state in {} steps.".format(max_steps))
    else:
        print("Episode reward: %f" % episode_reward)
    return episode_reward


def evaluate_policy(env, tabular_q, max_steps=100):
    """
    Print action for each state then evaluate the policy for one episode, no rendering
    :param env: environment
    :param tabular_q: q table
    :param max_steps: stop if the episode reaches max_steps
    :return: episode_reward: total reward for the episode.
    """

    episode_reward = 0
    actions_of_tabular_q(tabular_q)
    state, _ = env.reset()
    for t in range(max_steps):
        action = greedy_policy(tabular_q, state)
        state, reward, done, _, _ = env.step(action)
        episode_reward += reward
        if done:
            break
    if not done:
        print("The agent didn't reach a terminal state in {} steps.".format(max_steps))
    else:
        print("Episode reward: %f" % episode_reward)
    return episode_reward
