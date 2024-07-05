import torch
import torch.nn as nn


# Implements RNN Unit

class ExRNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(ExRNN, self).__init__()

        self.hidden_size = hidden_size
        self.sigmoid = torch.sigmoid

        # RNN Cell weights
        self.in2hidden = nn.Linear(input_size + hidden_size, hidden_size)
        # Hidden to output layer
        self.hidden2out = nn.Linear(hidden_size, output_size)

    def name(self):
        return "RNN"

    def forward(self, x, hidden_state):
        # Implementation of RNN cell
        hidden = self.sigmoid(self.in2hidden(torch.cat((x, hidden_state), dim=1)))
        output = self.sigmoid(self.hidden2out(hidden))

        return output, hidden

    def init_hidden(self, bs):
        return torch.zeros(bs, self.hidden_size)
