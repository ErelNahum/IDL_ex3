import torch
import numpy as np
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, input_size, representation_size):
        super(PositionalEncoding, self).__init__()

        pe = np.ones((input_size, representation_size))
        for pos in range(input_size):
            for i in range(0, representation_size, 2):
                arg = pos / 10000 ** (i / representation_size)
                pe[pos, i] = np.sin(arg)
                pe[pos, i + 1] = np.cos(arg)
        pe = torch.from_numpy(pe).float().unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x += self.pe[:x.size(0), :]
        return x
