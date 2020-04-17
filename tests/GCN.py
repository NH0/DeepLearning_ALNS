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

from ALNS.alns_state import CvrpState, generate_initial_solution
from ALNS.generate_instances import generate_cvrp_instance

EMBEDDING_SIZE = 15
OUTPUT_SIZE = 3
HIDDEN_SIZE = 128

MAX_EPOCH = 7000


class GCN(nn.Module):
    def __init__(self, input_features, hidden_size, output_feature):
        super(GCN, self).__init__()
        self.convolution1 = GraphConv(input_features, hidden_size)
        self.convolution2 = GraphConv(hidden_size, hidden_size)
        self.convolution3 = GraphConv(hidden_size, hidden_size)
        self.convolution4 = GraphConv(hidden_size, output_feature)

    def forward(self, graph, inputs):
        h = self.convolution1(graph, inputs)
        h = torch.relu(h)
        h = self.convolution2(graph, h)
        h = torch.relu(h)
        h = self.convolution3(graph, h)
        h = torch.relu(h)
        h = self.convolution4(graph, h)
        # Produce output of shape (input_size, output_features) instead of (input_size, *, output_features)
        h = torch.max(h, 1)[0]
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


def generate_karate_club_graph():
    nx_graph = nx.karate_club_graph()
    number_of_nodes = nx_graph.number_of_nodes()
    # nx.draw(nx_graph)
    # plt.show()
    degrees = [degree for index, degree in nx_graph.degree]
    dgl_graph = dgl.DGLGraph()
    dgl_graph.from_networkx(nx_graph=nx_graph)
    dgl_graph.ndata['demand'] = torch.tensor(
        [[np.random.random()] + [0] * (EMBEDDING_SIZE - 1) for node in range(number_of_nodes)])
    dgl_graph.ndata['isDepot'] = torch.tensor([[1.0] + [0] * (EMBEDDING_SIZE - 1)
                                               if np.random.randint(0, number_of_nodes) > number_of_nodes - 2
                                               else [0.0] + [0] * (EMBEDDING_SIZE - 1)
                                               for node in range(number_of_nodes)])

    return dgl_graph, degrees


def make_complete_graph(initial_state):
    number_of_nodes = initial_state.instance.number_of_nodes()
    edges_in_complete_graph = [(u, v) for u in range(number_of_nodes) for v in range(number_of_nodes) if u != v]
    for u, v in edges_in_complete_graph:
        initial_state.instance.add_edge(u, v, weight=initial_state.distances[u][v])


def create_cvrp_state():
    cvrp_instance = generate_cvrp_instance()
    # Create an empty state
    initial_state = CvrpState(cvrp_instance, collect_alns_statistics=False, seed=123456)
    # initial_solution = generate_initial_solution(initial_state)
    make_complete_graph(initial_state)

    return initial_state


def generate_cvrp_graph(nx_graph):
    number_of_nodes = nx_graph.number_of_nodes()
    # nx.draw(nx_graph)
    # plt.show()
    degrees = [degree for index, degree in nx_graph.degree]
    dgl_graph = dgl.DGLGraph()
    dgl_graph.from_networkx(nx_graph=nx_graph)
    dgl_graph.ndata['demand'] = torch.tensor(
        [[nx_graph.nodes[node]['demand']] + [0] * (EMBEDDING_SIZE - 1) for node in range(number_of_nodes)],
        dtype=torch.float)
    dgl_graph.ndata['isDepot'] = torch.tensor(
        [[nx_graph.nodes[node]['isDepot']] + [0] * (EMBEDDING_SIZE - 1) for node in range(number_of_nodes)],
        dtype=torch.float)
    dgl_graph.edata['weight'] = torch.tensor(
        [nx_graph.edges[u, v]['weight'] for u in range(number_of_nodes) for v in range(number_of_nodes) if u != v]
    )

    return dgl_graph, degrees


def generate_inputs(dgl_graph):
    number_of_nodes = dgl_graph.number_of_nodes()

    node_embedding = nn.Embedding(number_of_nodes, EMBEDDING_SIZE)
    dgl_graph.ndata['embedding'] = node_embedding.weight

    inputs = torch.stack([node_embedding.weight, dgl_graph.ndata['demand'], dgl_graph.ndata['isDepot']], dim=1)
    train_mask = torch.tensor([1 if np.random.randint(0, 4) > 0 else 0 for i in range(number_of_nodes)]).bool()

    return inputs, train_mask, node_embedding


def generate_labels(degrees):
    degrees_tensor = torch.tensor(degrees)
    labels = torch.tensor([[1.0 if i == degree else 0.0 for i in range(OUTPUT_SIZE)] for degree in degrees])

    return degrees_tensor, labels


def main():
    myGCN = GCN(EMBEDDING_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
    cvrp_state = create_cvrp_state()
    dgl_graph, degrees = generate_cvrp_graph(cvrp_state.instance)
    # dgl_graph, degrees = generate_karate_club_graph()
    inputs, train_mask, node_embedding = generate_inputs(dgl_graph)
    degrees_tensor, labels = generate_labels(degrees)

    optimizer = torch.optim.Adam(itertools.chain(myGCN.parameters(), node_embedding.parameters()), lr=0.0005)
    loss_function = nn.MSELoss()

    for epoch in range(MAX_EPOCH + 1):
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


if __name__ == '__main__':
    main()
# Epoch 10000, loss 0.0170, random loss 9.6572, accuracy 0.6250
# Epoch 10000, loss 0.0129, random loss 9.6366, accuracy 0.5000
# Epoch 10000, loss 0.0108, random loss 9.6581, accuracy 0.4545

# tensor([2, 2, 2, 2, 2, 2, 2, 1, 2, 1, 4, 1, 2, 2]) tensor([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
# Epoch 0, loss 0.0467, random loss 9.6502, accuracy 0.7143
# tensor([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]) tensor([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
# Epoch 1000, loss 0.0027, random loss 9.6355, accuracy 1.0000

# With edge weights
# tensor([19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19]) tensor([100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100])
# Epoch 0, loss 0.0025, random loss 9.2972, accuracy 0.0000
# tensor([19,  2,  2,  2,  2,  2,  8,  2, 13,  2,  2,  2,  2,  2]) tensor([100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100])
# Epoch 1000, loss 0.0025, random loss 9.2810, accuracy 0.0000
# tensor([19,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2]) tensor([100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100])
# Epoch 2000, loss 0.0025, random loss 9.2799, accuracy 0.0000
# tensor([19,  0,  0,  0,  0,  0,  8,  0, 13,  0,  0,  0,  0,  0]) tensor([100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100])
# Epoch 3000, loss 0.0025, random loss 9.2993, accuracy 0.0000
# tensor([19, 18, 18, 19, 18, 18, 18, 18, 18, 18, 18, 18, 19, 18]) tensor([100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100])
# Epoch 4000, loss 0.0025, random loss 9.2895, accuracy 0.0000
# tensor([ 8,  6,  6,  6, 13,  6,  8,  6, 13,  6,  6,  6,  6,  6]) tensor([100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100])
# Epoch 5000, loss 0.0025, random loss 9.2810, accuracy 0.0000
# tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]) tensor([100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100])
# Epoch 6000, loss 0.0025, random loss 9.3017, accuracy 0.0000
# tensor([19,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6]) tensor([100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100])
# Epoch 7000, loss 0.0025, random loss 9.3033, accuracy 0.0000
#
# Process finished with exit code 0
