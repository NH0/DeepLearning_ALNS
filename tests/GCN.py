#!/home/thib/Documents/Travail/CIRRELT/pythonLNS-env/bin/python
import itertools
import dgl
import torch
import os

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import networkx as nx

from dgl.nn.pytorch import GraphConv

from ALNS.alns_state import CvrpState
from ALNS.generate_instances import generate_cvrp_instance
from NeuralNetwork.create_dataset import retrieve_alns_stats

STATISTICS_DATA_PATH = os.getcwd().rpartition('/')[0] + '/data/'
ALNS_ITERATION_STATISTICS_FILE = '1inst_50nod_40cap_1dep_5000iter_0.8decay_0.35destr_18determ.pickle'
ALNS_ITERATION_STATISTICS_PATH = STATISTICS_DATA_PATH + ALNS_ITERATION_STATISTICS_FILE

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


def evaluate(network, inputs, degrees_tensor, train_mask):
    test_mask = ~train_mask
    network.eval()
    with torch.no_grad():
        correct = 0
        for graph, input_data in inputs:
            logits = network(graph, input_data)
            logits = logits[test_mask]
            degrees = degrees_tensor[test_mask]
            _, indices = torch.max(logits, dim=1)
            correct += torch.sum(indices == degrees).item()
        return correct / len(degrees)


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


def create_cvrp_state(size, number_of_depots, capacity, seed):
    cvrp_instance = generate_cvrp_instance(size=size, capacity=capacity, number_of_depots=number_of_depots, seed=seed)
    # Create an empty state
    initial_state = CvrpState(cvrp_instance, size=size, capacity=capacity, number_of_depots=number_of_depots,
                              collect_alns_statistics=False, seed=seed)
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
    # delete degrees from return value
    return dgl_graph, degrees


def generate_inputs(list_of_dgl_graphs, alns_iteration_statistics):
    # All graphs must have the same number of node
    # They actually represent the same CVRP problem with different solution
    # The data is the CVRP state at different iterations of the ALNS heuristic
    number_of_nodes = list_of_dgl_graphs[0].number_of_nodes()

    node_embedding = nn.Embedding(number_of_nodes, EMBEDDING_SIZE)

    inputs_data = []
    for i, graph in enumerate(list_of_dgl_graphs):
        graph.ndata['embedding'] = node_embedding.weight
        graph.ndata['isDeleted'] = torch.tensor(
            [[1 if node in alns_iteration_statistics['Statistics'][i]['destroyed_nodes'] else 0]
             + [0] * (EMBEDDING_SIZE - 1)
             for node in range(number_of_nodes)]
        )
        graph.edata['isSolution'] = torch.tensor(
            [1 if (u, v) in alns_iteration_statistics['Statistics'][i]['list_of_edges'] else 0
             for u in range(number_of_nodes) for v in range(number_of_nodes) if u != v]
        )
        inputs_data.append(torch.stack([node_embedding.weight, graph.ndata['demand'], graph.ndata['isDepot']], dim=1))
    inputs = list(zip(list_of_dgl_graphs, inputs_data))

    train_mask = torch.tensor([1 if np.random.randint(0, 4) > 0 else 0 for _ in range(number_of_nodes)]).bool()

    return inputs, train_mask, node_embedding


# to be modified
def generate_labels(degrees):
    degrees_tensor = torch.tensor(degrees)
    labels = torch.tensor([[1.0 if i == degree else 0.0 for i in range(OUTPUT_SIZE)] for degree in degrees])

    return degrees_tensor, labels


def main():
    graph_convolutional_network = GCN(EMBEDDING_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
    # Warning : can be a list of dictionaries, here considered to be a single dictionary
    alns_iteration_statistics = retrieve_alns_stats(ALNS_ITERATION_STATISTICS_PATH)
    if type(alns_iteration_statistics) != dict:
        print("Error, the stats file contains different CVRP instances.")
        return -1

    cvrp_state = create_cvrp_state(size=alns_iteration_statistics['Size'],
                                   number_of_depots=alns_iteration_statistics['Number_of_depots'],
                                   capacity=alns_iteration_statistics['Capacity'],
                                   seed=alns_iteration_statistics['Seed'])
    degrees = generate_cvrp_graph(cvrp_state.instance)[1]
    # dgl_graph, degrees = generate_karate_club_graph()

    list_of_dgl_graphs = [generate_cvrp_graph(cvrp_state.instance)[0]] * len(alns_iteration_statistics['Statistics'])
    inputs, train_mask, node_embedding = generate_inputs(list_of_dgl_graphs, alns_iteration_statistics)
    degrees_tensor, labels = generate_labels(degrees)

    optimizer = torch.optim.Adam(itertools.chain(graph_convolutional_network.parameters(), node_embedding.parameters()),
                                 lr=0.0005)
    loss_function = nn.MSELoss()

    epoch: int
    for epoch in range(MAX_EPOCH + 1):
        for graph, input_data in inputs:
            logits = graph_convolutional_network(graph, input_data)
            logp = F.softmax(logits, 1)
            loss = loss_function(logp[train_mask], labels[train_mask])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % 1000 == 0:
            random_logit = torch.tensor([[np.random.rand() for _ in range(OUTPUT_SIZE)] for _ in degrees])
            random_loss = loss_function(F.log_softmax(random_logit, 1)[train_mask], labels[train_mask])
            accuracy = evaluate(graph_convolutional_network, inputs, degrees_tensor, train_mask)
            print(
                'Epoch %d, loss %.4f, random loss %.4f, accuracy %.4f' % (
                    epoch, loss.item(), random_loss.item(), accuracy))


if __name__ == '__main__':
    main()
