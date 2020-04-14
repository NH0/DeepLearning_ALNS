#!/home/thib/Documents/Travail/CIRRELT/pythonLNS-env/bin/python
import itertools

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import dgl
import torch
from dgl.nn.pytorch import GraphConv

EMBEDDING_SIZE = 5
OUTPUT_SIZE = 2
HIDDEN_SIZE = 5

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


nx_graph = nx.karate_club_graph()
degrees = [degree for index, degree in nx_graph.degree]
dgl_graph = dgl.DGLGraph()
dgl_graph.from_networkx(nx_graph=nx_graph)
G = dgl_graph

net = GCN(EMBEDDING_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)

embed = nn.Embedding(34, 5)  # 34 nodes with embedding dim equal to 5
G.ndata['feat'] = embed.weight

inputs = embed.weight
labeled_nodes = torch.tensor([0, 33])  # only the instructor and the president nodes are labeled
labels = torch.tensor([0, 1])  # their labels are different

optimizer = torch.optim.Adam(itertools.chain(net.parameters(), embed.parameters()), lr=0.01)
for epoch in range(50):
    logits = net(G, inputs)
    # we save the logits for visualization later
    logp = F.log_softmax(logits, 1)
    # we only compute loss for labeled nodes
    loss = F.nll_loss(logp[labeled_nodes], labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print('Epoch %d | Loss: %.4f' % (epoch, loss.item()))


