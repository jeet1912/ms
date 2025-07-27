import argparse


def get_args():
    parser = argparse.ArgumentParser()

    # Method
    parser.add_argument('-method', type=str, choices=['DQN', 'test-CartPole', 'test-PyTorch'],
                        default='test-PyTorch', help='DQN or test CartPole')
    parser.add_argument('-reward_method', type=str, choices=['return', 'ratio', 'extreme'],
                        default='return', help='method to save the reward')

    # Parameters
    parser.add_argument('-seeds', type=int, default=1, help='random seeds, in range [0, seeds)')
    parser.add_argument('-epsilon', type=float, default=0.9, help='epsilon to control the convergence of iteration')
    parser.add_argument('-gamma', type=float, default=0.9, help='gamma for decreasing the reward')
    parser.add_argument('-torch_seed', type=int, default=0, help='random seeds for torch')

    # Network parameters
    parser.add_argument('-input_dim', type=int, default=2, help='input dimension for the network')
    parser.add_argument('-hidden_dim', type=int, default=10, help='hidden dimension for the network')
    parser.add_argument('-output_dim', type=int, default=1, help='output dimension for the network')
    parser.add_argument('-num_hidden', type=int, default=0, help='number of hidden layers for the network')
    parser.add_argument('-lr', type=float, default=0.001, help='learning rate for the network')

    # DQN parameters
    parser.add_argument('-batch_size', type=int, default=60, help='batch size')
    parser.add_argument('-memory_capacity', type=int, default=500, help='memory capacity')
    parser.add_argument('-target_replace_iter', type=int, default=10, help='target replace iteration')
    parser.add_argument('-target_update_method', type=str, default='hard', choices=['hard', 'soft'],
                        help="hard or soft update for the target network")
    parser.add_argument('-soft_update_tau', type=float, default=0.01,
                        help='the ratio of the target network parameters when using the soft update method')

    # Render mode
    parser.add_argument(
        "-render_mode",
        "-r",
        type=str,
        help="The render mode for the environment. 'human' opens a window to render. 'ansi' does not render anything.",
        choices=["human", "ansi"],
        default="human",
    )

    return parser.parse_args()
