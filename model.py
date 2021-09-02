import numpy as np
from torch import nn


class Model(nn.Module):

    def __init__(self, dump: bool = False):
        super(Model, self).__init__()
        self.stack = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.MaxPool2d((2, 2)),
            nn.Dropout2d(),
            nn.Flatten(),
            nn.ReLU(),
            nn.Linear(1024, 3200, bias=True),
            nn.ReLU(),
            nn.Linear(3200, 26, bias=True)
        )

    def forward(self, x):
        return self.stack(x)
