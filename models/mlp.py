import torch.nn as nn

from models.matmul import MatMul


class ExMLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(ExMLP, self).__init__()

        self.ReLU = nn.ReLU()

        # Token-wise MLP network weights
        self.layer1 = MatMul(input_size, hidden_size)
        # additional layer(s)
        self.layers = nn.Sequential(
            MatMul(input_size, hidden_size),
            nn.ReLU(),
            MatMul(hidden_size, output_size),
        )

    def name(self):
        return "MLP"

    def forward(self, x):
        # Token-wise MLP network implementation
        return self.layers(x)
