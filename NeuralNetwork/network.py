import torch
import torch.nn as nn
import numpy as np
import sys

from NeuralNetwork.create_dataset import create_dataset

HIDDEN_SIZE = 512

if torch.cuda.is_available():
    print("yes")


# class RnnModule(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size):
#         super(RnnModule, self).__init__()
#
#         self.hidden_size = hidden_size
#         self.input_size = input_size
#         self.output_size = output_size
#
#         self.Wx = nn.Linear(hidden_size[0], input_size[1])
#         self.Wh = nn.Linear(hidden_size[1], hidden_size[0])
#         self.Wy = nn.Linear(output_size[0], hidden_size[1])
#         self.bh = torch.randn(hidden_size)
#         self.by = torch.randn(output_size)
#
#     def forward(self, x, h):
#         next_h = nn.LogSoftmax()(self.Wh(h) + self.Wx(x) + self.bh)
#         output = self.Wy(next_h)
#         return output, next_h

if __name__ == '__main__':
    if len(sys.argv) == 2:
        x, y = create_dataset(sys.argv[1])
        input_size = len(x[0][0])
        sequence_length = len(x[0])
        rnn = nn.RNN(input_size, HIDDEN_SIZE)
        h = torch.randn(1, sequence_length, HIDDEN_SIZE)
        x_input = torch.tensor(x[0])
        x_input = torch.reshape(x_input, (sequence_length, sequence_length, input_size))
        output, next_h = rnn(x_input, h)
        print(output, next_h)
