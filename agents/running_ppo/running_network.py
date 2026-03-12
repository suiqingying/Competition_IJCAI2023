import torch
import torch.nn as nn
import torch.nn.functional as F

class RunningCNN_Actor(nn.Module):
    def __init__(self, action_space):
        super(RunningCNN_Actor, self).__init__()
        # Input: (Batch, 3, 40, 40)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=4, stride=2)   # (16, 19, 19)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2)  # (32, 9, 9)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1)  # (64, 7, 7)
        self.bn3 = nn.BatchNorm2d(64)
        
        self.fc1 = nn.Linear(64 * 7 * 7, 256)
        self.fc2 = nn.Linear(256, action_space)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        action_prob = F.softmax(self.fc2(x), dim=-1)
        return action_prob
