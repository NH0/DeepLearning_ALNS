import torch.nn as nn
import torch.nn.functional as F


class MLPReadout(nn.Module):

    def __init__(self, input_dim, output_dim, L=2):  # L=nb_hidden_layers
        super().__init__()
        list_FC_layers = [nn.Linear(input_dim // 2 ** layer, input_dim // 2 ** (layer + 1), bias=True)
                          for layer in range(L)]
        list_FC_layers.append(nn.Linear(input_dim // 2 ** L, output_dim, bias=True))
        self.FC_layers = nn.ModuleList(list_FC_layers)
        self.L = L

    def forward(self, x):
        y = x
        for layer in range(self.L):
            y = self.FC_layers[layer](y)
            y = F.relu(y)
        y = self.FC_layers[self.L](y)
        return y
