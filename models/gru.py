import torch
import torch.nn as nn

# Implements GRU Unit

class ExGRU(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(ExGRU, self).__init__()
        self.hidden_size = hidden_size
        self.sigmoid = torch.sigmoid

        # GRU Cell weights
        self.update_gate_layer = nn.Linear(hidden_size + input_size, hidden_size)
        self.reset_gate_layer = nn.Linear(hidden_size + input_size, hidden_size)
        self.hidden_activation_layer = nn.Sequential(
            nn.Linear(hidden_size + input_size, hidden_size),
            nn.Tanh()
        )

        self.hidden2out = nn.Linear(hidden_size, output_size)

    def name(self):
        return "GRU"

    def forward(self, x, hidden_state):
        # Implementation of GRU cell
        concatenated = torch.cat((hidden_state, x), dim=1)

        zt = self.sigmoid(self.update_gate_layer(concatenated))
        rt = self.sigmoid(self.reset_gate_layer(concatenated))
        h_tilda = self.hidden_activation_layer(torch.cat((rt * hidden_state, x), dim=1))
        hidden = (1 - zt) * hidden_state + zt * h_tilda

        output = self.hidden2out(hidden)

        return output, hidden

    def init_hidden(self, bs):
        return torch.zeros(bs, self.hidden_size)
