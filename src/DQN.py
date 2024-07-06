import torch
import torch.nn as nn
import numpy as np


class DQN(nn.Module):

    def __init__(self, input_shape: np.array, num_actions: int) -> None:
        """
        Args:
            input_shape (np.array): input shape of the environment
            num_actions (int): action space size
        """
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2),
            nn.ReLU(),
        )

        conv_feature_size = self._calculate_feature_size(input_shape)

        self.fc = nn.Sequential(
            nn.Linear(conv_feature_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions),
        )

    def _calculate_feature_size(self, input_shape) -> int:
        """Calculate the output size of the convolutional layer

        Args:
            input_shape (np.array): input shape of the environment

        Returns:
            int: output size of the convolutional layer
        """
        output = self.conv(torch.zeros(1, *input_shape))
        return int(np.prod(output.size()[1:]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the network

        Args:
            x (torch.Tensor): input tensor

        Returns:
            torch.Tensor: output tensor
        """
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
