import torch
from torch import nn
import torch.nn.functional as F

# Define the model
class CNNActionValue(nn.Module):
    def __init__(self, state_dim, action_dim, activation=F.relu):
        super(CNNActionValue, self).__init__()
        self.conv1 = nn.Conv2d(state_dim, 16, kernel_size=8, stride=4) #input: 84x84 image with 4 channels (4 stacked frames)
        # 16 different 8x8 filters, each jumping 4 pixels
        # output: 20x20 with 16 channels (one per filter)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        # input: 20x20 with 16 channels 
        # 32 different 4x4 filters, jumping 2 pixels
        # output: 9x9 with 32 channels
        self.in_features = 32 * 9 * 9 
        self.fc1 = nn.Linear(self.in_features, 256)
        self.fc2 = nn.Linear(256, action_dim)  # output layer

    def forward(self, x):
        x = F.relu(self.fc1(x))  # apply ReLu (rectified linear unit) activation
        x = self.out(x)  # calculate output
        return x