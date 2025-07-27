from util import plot_episode_rewards
# uncomment the following two lines if you are using wayland(Ubuntu) as the backend. Comment if not using wayland
import os
os.environ["QT_QPA_PLATFORM"] = "wayland"
import random

def init_states(env):
    """
    reset the env and print the initial state 3 times
    :param env: cartpole environment
    :return:
    """
    for i in range(3):
        state, _ = env.reset()
        print(f'state:{state}')


def test_each_action(env):
    """
    reset the env and test each action
    :param env: cartpole environment
    :return:
    """
    state, _ = env.reset()
    for action in range(env.action_space.n):
        #############################
        # Your Code #
        # Call the step function to test each action #
        # save the next state, reward, done, truncated and print

        next_state, reward, done, truncated, _ = env.step(action)
        #############################
        print(f'state:{state}, action:{action}, reward:{reward}, done:{done}, truncated:{truncated}, '
              f'next_state:{next_state}')
        state = next_state


def test_moves(env, policy='random'):
    """
    reset the env and test the policy
    :param env: cliff walking environment
    :param actions: a list of actions to act on the environment
    :return:
    """
    # episode_rewards: a list to store the total reward for each episode
    episode_rewards = []
    # candidate_actions can be selected
    candidate_actions = [0, 1]
    for _ in range(1000):
        total_reward = 0  # initialize the total reward for each episode
        state, _ = env.reset()
        done = truncated = False

        while not (done or truncated):
            #############################
            # Your Code #
            # implement the policy to move the agent
            # If policy is 'random', randomly select an action from candidate_actions (left or right).
            #       You can use random.sample to select an action from candidate_actions
            # If policy is 'right', always select action as pulling the cart to the right regardless of the state
            # If policy is 'left', always select action as pulling the cart to the left regardless of the state
            # Please read the gym documentation to find the representation each action
            # Take the action and save the next state, reward, done, truncated
            if policy == 'random':
                action = random.sample(candidate_actions,1)[0]
                #print(action)
            elif policy == 'right':
                action = 1
            elif policy == 'left':
                action = 0
            else: 
                raise ValueError("Unkown policy")

            next_state, reward, done, truncated, _ = env.step(action)
            #############################
            total_reward += reward
            state = next_state

        episode_rewards.append(total_reward)

    plot_episode_rewards(episode_rewards, f'episode_rewards_policy:{policy}.png')
