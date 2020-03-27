import torch
import torch.nn as nn
import numpy as np

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


rnn = nn.RNN(3, 5)
x = torch.tensor([[[30,1,2], [43,5,2], [38,1,0], [7, 3, 0], [3, 0, 0], [48, 2, 2]]], dtype=torch.float)
h = torch.zeros(1, 6, 5)
print(x.size())
output, next_h = rnn(x, h)
print(output, next_h)
