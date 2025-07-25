import gymnasium as gym


def test_each_action(env):
    """
    reset the env and test each action
    :param env: cliff walking environment
    :return:
    """
    state, _ = env.reset()
    print(state)
    for action in range(env.action_space.n):
        state, _ = env.reset()
        next_state, reward, done, _, _ = env.step(action)
        print(f'state:{state}, action:{action}, reward:{reward}, done:{done}, next_state:{next_state}')


def test_moves(env, actions):
    """
    reset the env and test the policy
    :param env: cliff walking environment
    :param actions: a list of actions to act on the environment
    :return:
    """
    total_reward = 0
    ############################
    # Your Code #
    # Imitate the test_each_action function, take the action one by one and move to the destination state #
    # You need to call the env.step to get the next state, reward, and other information #
    # Please print the state, reward, and done for each step #
    # I b
    state, _ = env.reset()
    for action in actions:
        nextState, reward, done, _, _ = env.step(action)
        print(f'state :{state}, action:{action}, reward:{reward}, done:{done}, next_state:{nextState}')
        state = nextState
        total_reward += reward



    ############################
    print(f'total reward:{total_reward}')

