import torch
import numpy as np
import torch.nn as nn
from torch.nn.functional import pad

from models.matmul import MatMul
from models.positional_encoding import PositionalEncoding

atten_size = 5


class ExRestSelfAtten(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, atten_size):
        super(ExRestSelfAtten, self).__init__()

        self.input_size = input_size
        self.atten_size = atten_size
        self.sqrt_hidden_size = np.sqrt(float(hidden_size))
        self.ReLU = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(2)
        self.positional_encode = PositionalEncoding(input_size, input_size)

        # Token-wise MLP + Restricted Attention network implementation

        self.layer1 = nn.Sequential(
            MatMul(input_size, hidden_size),
            nn.ReLU()
        )
        self.W_q = MatMul(hidden_size, hidden_size, use_bias=False)
        self.W_k = MatMul(hidden_size, hidden_size, use_bias=False)
        self.W_v = MatMul(hidden_size, hidden_size, use_bias=False)

        self.atten2out = MatMul(hidden_size, output_size)

    def name(self):
        return "MLP_atten"

    def forward(self, x):
        x = self.positional_encode(x)

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

        # Applying attention layer
        query = self.W_q(x)  # TODO: if it doesnt work, unsqueeze(2) and fix einsum
        keys = self.W_k(x_nei)
        vals = self.W_v(x_nei)

        atten_weights = torch.einsum('bwh,bwoh->bwo', [query, keys]) / self.sqrt_hidden_size
        atten_weights = self.softmax(atten_weights)

        x = torch.einsum('bwo,bwoh->bwh', [atten_weights, vals])

        x = self.atten2out(x)

        return x, atten_weights
