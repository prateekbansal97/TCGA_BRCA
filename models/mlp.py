import torch.nn as nn

class MLP(nn.Module):                                               #MLP with ReLU and Dropout
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),                                           #Sigmoid Activation
        )

    def forward(self, x):
        return self.model(x)
