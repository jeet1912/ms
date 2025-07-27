import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


class Net(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_hidden):
        """
        Define the network
        :param input_dim: the dimension of input
        :param hidden_dim: the dimension of hidden layer
        :param output_dim: the dimension of output
        :param num_hidden: the number of hidden layers.
        """
        super(Net, self).__init__()
        ############################
        # Define the network.
        # The network include one input layer, num_hidden hidden layers, and one output layer.

        self.num_hidden = num_hidden
        # Define #1 input layer
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        # Define #num_hidden hidden layers
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim)
                                            for _ in range(self.num_hidden)])
        # Define #1 output layer
        self.output_layer = nn.Linear(hidden_dim, output_dim)



        ############################

    def forward(self, x):
        ############################
        # Define the forward pass.

        # The input is passed by the input layer
        x = self.input_layer(x)
        # Activation function is applied
        x = F.relu(x)
        # Hidden layers are applied.
        for layer in self.hidden_layers:
            x = layer(x)
            x = F.relu(x)
        # Output is obtained by the output layer
        output = self.output_layer(x)



        ############################
        return output


class Practice(object):
    def __init__(self, input_dim=2, hidden_dim=10, output_dim=1, num_hidden=2, lr=0.1):
        """
        Define the network, optimizer, and loss function. Train the network
        :param input_dim: the dimension of input
        :param hidden_dim: the dimension of hidden layer
        :param output_dim: the dimension of output
        :param num_hidden: the number of hidden layers. Hidden layers include the output layer
        :param lr: the learning rate of the optimizer
        """
        # Define the network
        self.net = Net(input_dim, hidden_dim, output_dim, num_hidden)
        # device: use GPU if available, else use CPU
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        ############################
        # Move the net to the device
        self.net.to(self.device)
        # Optimizer: use Adam, set the learning rate as lr
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        # Define loss function as MSELoss
        self.loss_function = nn.MSELoss()



        ############################

    def train(self, x, y):
        """
        Train the network for 1000 epochs
        :param x: input data
        :param y: output data
        :return: record_loss: the loss of each epoch
        """
        record_loss = []  # record the loss
        ############################
        # Move x and y to the device
        x = x.to(self.device)
        y = y.to(self.device)



        ############################

        for t in range(1000):
            ############################
            # Pass the x through the network.
            output = self.net(x)
            # Calculate the loss and define the variable as loss
            loss = self.loss_function(output, y)
            # Clear the gradient
            self.optimizer.zero_grad()
            # Backpropagation on the loss function
            loss.backward()
            # Update the weights
            self.optimizer.step()
            # Store the loss for each cycle
            record_loss.append(loss.item())
        ############################
        return record_loss


def plot_loss(loss_list, lr, hidden_dim, num_hidden):
    """
    Plot the loss. A good implementation will have a gradually decreasing loss
    :param loss_list: list of loss
    :param lr: learning rate
    :param hidden_dim: hidden dimension
    :param num_hidden: number of hidden layers
    :return: save the plot to a file
    """
    plt.plot(loss_list)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.ylim(0, 0.5)
    plt.title(f'min loss:{min(loss_list)}')
    plt.savefig(f'loss_learningRate:{lr}_hiddenDim:{hidden_dim}_numHidden:{num_hidden}.png')
    # plt.show()


def test_torch_NN(args):
    # define the input and output. The relationship between x and y is y = x1 AND x2
    x = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
    y = torch.tensor([[0], [0], [0], [1]], dtype=torch.float32)

    # set the torch seed
    torch.manual_seed(args.torch_seed)
    # define the network
    practice_net = Practice(args.input_dim, args.hidden_dim, args.output_dim, args.num_hidden, args.lr)
    # train the network and get the loss of each epoch
    loss_list = practice_net.train(x, y)
    # plot the loss
    plot_loss(loss_list, args.lr, args.hidden_dim, args.num_hidden)

