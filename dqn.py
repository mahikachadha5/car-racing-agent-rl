import torch
from torch import nn
import torch.nn.functional as F

# Define the model
class DQN(nn.Module):
    def __init__(self, in_states, hl_nodes, out_actions):
        super().__init__()

        # define network layers
        self.fc1 = nn.Linear(in_states, hl_nodes)  # first fully connected layer
        self.out = nn.Linear(out_actions, hl_nodes)  # output layer

    def forward(self, x):
        x = F.relu(self.fc1(x))  # apply ReLu (rectified linear unit) activation
        x = self.out(x)  # calculate output
        return x
