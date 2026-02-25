import torch
from torch import nn
import torch.nn.functional as F


class CNNActionValue(nn.Module):
    def __init__(self, state_dim, action_dim, activation=F.relu):
        super(CNNActionValue, self).__init__()
        # input: 84x84 image with 4 channels (4 stacked frames)
        # 16 different 8x8 filters, each jumping 4 pixels
        # output: 20x20 with 16 channels
        self.conv1 = nn.Conv2d(state_dim, 16, kernel_size=8, stride=4)
        # input: 20x20 with 16 channels
        # 32 different 4x4 filters, jumping 2 pixels
        # output: 9x9 with 32 channels
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.in_features = 32 * 9 * 9  # flatten
        self.fc1 = nn.Linear(self.in_features, 256)
        self.fc2 = nn.Linear(256, action_dim)
        self.activation = activation

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view((-1, self.in_features))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
