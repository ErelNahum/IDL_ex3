import torch
import numpy as np
import torch.nn as nn
from torch.nn.functional import pad

from models.matmul import MatMul

atten_size = 5


class ExRestSelfAtten(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, atten_size):
        super(ExRestSelfAtten, self).__init__()

        self.input_size = input_size
        self.atten_size = atten_size
        self.sqrt_hidden_size = np.sqrt(float(hidden_size))
        self.ReLU = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(2)

        # Token-wise MLP + Restricted Attention network implementation

        self.layer1 = nn.Sequential(
            MatMul(input_size, hidden_size),
            nn.ReLU()
        )
        self.W_q = MatMul(hidden_size, hidden_size, use_bias=False)
        self.W_k = MatMul(hidden_size, hidden_size, use_bias=False)
        self.W_v = MatMul(hidden_size, hidden_size, use_bias=False)

        self.atten2out = MatMul(hidden_size, output_size)
        # rest ...

    def name(self):
        return "MLP_atten"

    def forward(self, x):
        # Token-wise MLP + Restricted Attention network implementation
        x = self.layer1(x)

        # generating x in offsets between -atten_size and atten_size
        # with zero padding at the ends
        padded = pad(x, (0, 0, self.atten_size, self.atten_size, 0, 0))

        x_nei = []
        for k in range(-self.atten_size, self.atten_size + 1):
            x_nei.append(torch.roll(padded, k, 1))

        x_nei = torch.stack(x_nei, 2)
        x_nei = x_nei[:, self.atten_size:-self.atten_size, :]

        # x_nei has an additional axis that corresponds to the offset

        sub = []
        weights = []
        for i in range(2 * self.atten_size + 1):
            xi = x_nei[:, :, i, :]
            # Applying attention layer
            query = self.W_q(xi)
            keys = self.W_k(xi)
            vals = self.W_v(xi)

            weights.append(self.softmax(torch.bmm(query, keys.transpose(1, 2)) / self.sqrt_hidden_size))

            sub.append(torch.bmm(weights[-1], vals))

        x = torch.stack(sub, 2)
        x = self.atten2out(x)

        return x, torch.stack(weights, 2)
