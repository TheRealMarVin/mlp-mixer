import torch
import torch.nn as nn
import torch.nn.functional as  F

class ChannelMixer(nn.Module):
    def __init__(self, input_size, hidden_size, nb_channels, dropout=None):
        super(ChannelMixer, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, input_size)

        self.dropout = None
        if dropout is not None:
            self.dropout = nn.Dropout(dropout)

        self.activation = nn.GELU()

    def forward(self, x):
        input = x

        x = self.fc1(x)
        x = self.activation(x)

        if self.dropout is not None:
            x = self.dropout(x)

        x = self.fc2(x)
        x = self.activation(x)
        if self.dropout is not None:
            x = self.dropout(x)

        x = x + input
        return x
