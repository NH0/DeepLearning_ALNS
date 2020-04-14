#!/home/thib/Documents/Travail/CIRRELT/pythonLNS-env/bin/python
import itertools

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import networkx as nx
import dgl
import torch

from dgl.nn.pytorch import GraphConv

EMBEDDING_SIZE = 15
OUTPUT_SIZE = 20
HIDDEN_SIZE = 128


class GCN(nn.Module):
    def __init__(self, input_features, hidden_size, output_feature):
        super(GCN, self).__init__()
        self.convolution1 = GraphConv(input_features, hidden_size)
        self.convolution2 = GraphConv(hidden_size, hidden_size)
        self.convolution3 = GraphConv(hidden_size, output_feature)

    def forward(self, graph, inputs):
        h = self.convolution1(graph, inputs)
        h = torch.relu(h)
        h = self.convolution2(graph, h)
        h = torch.relu(h)
        h = self.convolution3(graph, h)
        return h


def evaluate(network, graph, inputs, degrees_tensor, train_mask):
    test_mask = ~train_mask
    network.eval()
    with torch.no_grad():
        logits = network(graph, inputs)
        logits = logits[test_mask]
        degrees = degrees_tensor[test_mask]
        _, indices = torch.max(logits, dim=1)
        print(indices, degrees)
        correct = torch.sum(indices == degrees)
        return correct.item() / len(degrees)


nx_graph = nx.karate_club_graph()
# nx.draw(nx_graph)
# plt.show()
number_of_nodes = nx_graph.number_of_nodes()
degrees = [degree for index, degree in nx_graph.degree]
dgl_graph = dgl.DGLGraph()
dgl_graph.from_networkx(nx_graph=nx_graph)

myGCN = GCN(EMBEDDING_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)

node_embedding = nn.Embedding(number_of_nodes, EMBEDDING_SIZE)
dgl_graph.ndata['embedding'] = node_embedding.weight
dgl_graph.ndata['demand'] = torch.tensor([np.random.random() for node in range(number_of_nodes)])
dgl_graph.ndata['isDepot'] = torch.tensor([1
                                           if np.random.randint(0, number_of_nodes) > number_of_nodes - 2
                                           else 0
                                           for node in range(number_of_nodes)])
inputs = node_embedding.weight
degrees_tensor = torch.tensor(degrees)
labels = torch.tensor([[1.0 if i == degree else 0.0 for i in range(OUTPUT_SIZE)] for degree in degrees])

optimizer = torch.optim.Adam(itertools.chain(myGCN.parameters(), node_embedding.parameters()), lr=0.0005)
loss_function = nn.MSELoss()

train_mask = torch.tensor([1 if np.random.randint(0, 4) > 0 else 0 for node in nx_graph.nodes]).bool()
for epoch in range(10001):
    logits = myGCN(dgl_graph, inputs)
    logp = F.softmax(logits, 1)
    loss = loss_function(logp[train_mask], labels[train_mask])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 1000 == 0:
        random_logit = torch.tensor([[np.random.rand() for i in range(OUTPUT_SIZE)] for degree in degrees])
        random_loss = loss_function(F.log_softmax(random_logit, 1)[train_mask], labels[train_mask])
        accuracy = evaluate(myGCN, dgl_graph, inputs, degrees_tensor, train_mask)
        print(
            'Epoch %d, loss %.4f, random loss %.4f, accuracy %.4f' % (epoch, loss.item(), random_loss.item(), accuracy))
