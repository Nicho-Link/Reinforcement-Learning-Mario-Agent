from torch import nn
import copy

class SMBAgentNN(nn.Module):
    def __init__(self, state_shape, num_actions):
        super().__init__()
        frames, height, width = state_shape

        if height != 240 or width != 256 or frames != 10:
            raise ValueError(f"Expecting state shape: (1, 240, 256, 10), got: {state_shape}")
        
        self.online = nn.Sequential(
            nn.Conv2d(in_channels=frames, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1600, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions),
        )
        self.target = copy.deepcopy(self.online)
        for param in self.target.parameters():
            param.requires_grad = False

    def forward(self, state, model):
        if model == "online":
            return self.online(state)
        elif model == "target":
            return self.target(state)
