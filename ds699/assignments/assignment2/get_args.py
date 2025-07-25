import argparse


def get_args():
    parser = argparse.ArgumentParser()

    # Render mode
    parser.add_argument(
        "-render_mode",
        "-r",
        type=str,
        help="The render mode for the environment. 'human' opens a window to render. 'ansi' does not render anything.",
        choices=["human", "ansi"],
        default="human",
    )

    # Method
    parser.add_argument('-method', type=str, choices=['test-cliff-walking', 'sarsa', 'q-learning'],
                        default='test-cliff-walking', help='select to run test-cliff-walking, sarsa or q-learning')

    # Parameters
    parser.add_argument('-seeds', type=int, default=1, help='random seeds, in range [0, seeds-1]')
    parser.add_argument('-init_epsilon', type=float, default=0.1,
                        help='initial epsilon to control the convergence of iteration')
    parser.add_argument('-gamma', type=float, default=0.9, help='gamma for decreasing the reward')
    parser.add_argument('-alpha', type=float, default=0.5, help='alpha for update the q table')
    parser.add_argument('-num_steps', type=int, default=1, help='number of steps for n-step sarsa')
    parser.add_argument('-num_episode', type=int, default=1000, help='number of episodes to update the q table')


    # Initializations
    parser.add_argument('-init_q_value', type=int, default=0, help='initial value for q table')

    return parser.parse_args()

