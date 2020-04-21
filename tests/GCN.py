#!/home/thib/Documents/Travail/CIRRELT/pythonLNS-env/bin/python
import itertools
import dgl
import torch
import os

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx

from dgl.nn.pytorch import GraphConv

from ALNS.alns_state import CvrpState
from ALNS.generate_instances import generate_cvrp_instance
from NeuralNetwork.create_dataset import retrieve_alns_stats

STATISTICS_DATA_PATH = os.getcwd().rpartition('/')[0] + '/data/'
ALNS_STATISTICS_FILE = '1inst_50nod_40cap_1dep_5000iter_0.8decay_0.35destr_18determ.pickle'

EMBEDDING_SIZE = 15
OUTPUT_SIZE = 3
HIDDEN_SIZE = 128

MAX_EPOCH = 1000

EPSILON = 1e-5


class GCN(nn.Module):
    def __init__(self, input_features, hidden_size, output_feature):
        super(GCN, self).__init__()
        self.convolution1 = GraphConv(input_features, hidden_size)
        self.convolution2 = GraphConv(hidden_size, hidden_size)
        self.convolution3 = GraphConv(hidden_size, hidden_size)
        self.convolution4 = GraphConv(hidden_size, hidden_size)
        self.linear = nn.Linear(hidden_size, output_feature)

    def forward(self, graph, inputs):
        h = self.convolution1(graph, inputs)
        h = torch.relu(h)
        h = self.convolution2(graph, h)
        h = torch.relu(h)
        h = self.convolution3(graph, h)
        h = torch.relu(h)
        h = self.convolution4(graph, h)
        # Produce output of shape (N, hidden_size) instead of (N, *, hidden_size)
        h = torch.max(h, 1)[0]
        # Return a tensor of shape (hidden_size)
        h = torch.mean(h, 0)
        h = self.linear(h)
        return h


def evaluate(network, inputs_test, labels, train_mask):
    test_mask = ~train_mask
    network.eval()
    with torch.no_grad():
        correct = 0
        for index, (graph, input_data) in enumerate(inputs_test):
            logits = network(graph, input_data)
            logp = F.softmax(logits, dim=0)
            # torch.max -> (max, argmax), so we only keep the argmax
            predicted_class = torch.argmax(logp, dim=0).item()
            true_class = torch.argmax(labels[test_mask][index], dim=0).item()
            correct += predicted_class == true_class
        return correct / len(test_mask)


def generate_karate_club_graph(embedding_size=EMBEDDING_SIZE):
    nx_graph = nx.karate_club_graph()
    number_of_nodes = nx_graph.number_of_nodes()
    # nx.draw(nx_graph)
    # plt.show()
    degrees = [degree for index, degree in nx_graph.degree]
    dgl_graph = dgl.DGLGraph()
    dgl_graph.from_networkx(nx_graph=nx_graph)
    dgl_graph.ndata['demand'] = torch.tensor(
        [[np.random.random()] + [0] * (embedding_size - 1) for _node in range(number_of_nodes)])
    dgl_graph.ndata['isDepot'] = torch.tensor([[1.0] + [0] * (embedding_size - 1)
                                               if np.random.randint(0, number_of_nodes) > number_of_nodes - 2
                                               else [0.0] + [0] * (embedding_size - 1)
                                               for _node in range(number_of_nodes)])

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


def generate_cvrp_graph(nx_graph, embedding_size=EMBEDDING_SIZE):
    number_of_nodes = nx_graph.number_of_nodes()
    # nx.draw(nx_graph)
    # plt.show()
    dgl_graph = dgl.DGLGraph()
    dgl_graph.from_networkx(nx_graph=nx_graph)
    dgl_graph.ndata['demand'] = torch.tensor(
        [[nx_graph.nodes[node]['demand']] + [0] * (embedding_size - 1) for node in range(number_of_nodes)],
        dtype=torch.float)
    dgl_graph.ndata['isDepot'] = torch.tensor(
        [[nx_graph.nodes[node]['isDepot']] + [0] * (embedding_size - 1) for node in range(number_of_nodes)],
        dtype=torch.float)
    dgl_graph.edata['weight'] = torch.tensor(
        [nx_graph.edges[u, v]['weight'] for u in range(number_of_nodes) for v in range(number_of_nodes) if u != v]
    )

    return dgl_graph


def generate_inputs(list_of_dgl_graphs, alns_instance_statistics, embedding_size=EMBEDDING_SIZE):
    # All graphs must have the same number of node
    # They actually represent the same CVRP problem with different solution
    # The data is the CVRP state at different iterations of the ALNS heuristic
    number_of_nodes = list_of_dgl_graphs[0].number_of_nodes()

    node_embedding = nn.Embedding(number_of_nodes, embedding_size)

    inputs_data = []
    for i, graph in enumerate(list_of_dgl_graphs):
        graph.ndata['embedding'] = node_embedding.weight
        graph.ndata['isDeleted'] = torch.tensor(
            [[1 if node in alns_instance_statistics['Statistics'][i]['destroyed_nodes'] else 0]
             + [0] * (embedding_size - 1)
             for node in range(number_of_nodes)],
            dtype=torch.float
        )
        graph.edata['isSolution'] = torch.tensor(
            [1 if (u, v) in alns_instance_statistics['Statistics'][i]['list_of_edges'] else 0
             for u in range(number_of_nodes) for v in range(number_of_nodes) if u != v]
        )
        inputs_data.append(torch.stack([node_embedding.weight,
                                        graph.ndata['demand'],
                                        graph.ndata['isDepot'],
                                        graph.ndata['isDeleted']], dim=1))
    inputs = list(zip(list_of_dgl_graphs, inputs_data))
    inputs_train = []
    inputs_test = []
    train_mask = []
    for index, single_input in enumerate(inputs):
        if np.random.randint(0, 4) > 0:
            inputs_train.append(single_input)
            train_mask.append(1)
        else:
            inputs_test.append(single_input)
            train_mask.append(0)

    train_mask = torch.tensor(train_mask).bool()

    return inputs_train, inputs_test, train_mask, node_embedding


def generate_labels(alns_instance_statistics, epsilon=EPSILON):
    labels = torch.tensor([[1, 0, 0] if iteration['objective_difference'] > 0
                           else [0, 1, 0] if abs(iteration['objective_difference']) <= epsilon
                           else [0, 0, 1]
                           for iteration in alns_instance_statistics['Statistics']],
                          dtype=torch.float)

    return labels


def main(alns_statistics_file=ALNS_STATISTICS_FILE,
         embedding_size=EMBEDDING_SIZE, hidden_size=HIDDEN_SIZE, output_size=OUTPUT_SIZE,
         max_epoch=MAX_EPOCH, epsilon=EPSILON):

    step = 1

    graph_convolutional_network = GCN(embedding_size, hidden_size, output_size)
    print("{0} Created GCN".format(step))
    step += 1

    alns_statistics_path = STATISTICS_DATA_PATH + alns_statistics_file
    # Warning : can be a list of dictionaries, here considered to be a single dictionary
    alns_instance_statistics = retrieve_alns_stats(alns_statistics_path)
    if type(alns_instance_statistics) != dict:
        print("Error, the stats file contains different CVRP instances.")
        return -1
    print("{0} Retrieved alns statistics".format(step))
    step += 1

    cvrp_state = create_cvrp_state(size=alns_instance_statistics['Size'],
                                   number_of_depots=alns_instance_statistics['Number_of_depots'],
                                   capacity=alns_instance_statistics['Capacity'],
                                   seed=alns_instance_statistics['Seed'])
    print("{0} Created new cvrp state".format(step))
    step += 1

    list_of_dgl_graphs =\
        [generate_cvrp_graph(cvrp_state.instance, embedding_size)] * len(alns_instance_statistics['Statistics'])
    inputs_train, inputs_test, train_mask, node_embedding = generate_inputs(list_of_dgl_graphs,
                                                                            alns_instance_statistics,
                                                                            embedding_size)
    labels = generate_labels(alns_instance_statistics, epsilon)
    print("{0} Created inputs and labels".format(step))
    step += 1

    optimizer = torch.optim.Adam(itertools.chain(graph_convolutional_network.parameters(), node_embedding.parameters()),
                                 lr=0.0005)
    loss_function = nn.MSELoss()

    number_of_iterations = len(alns_instance_statistics['Statistics'])
    number_of_null_iterations = 0
    for iteration in alns_instance_statistics['Statistics']:
        if abs(iteration['objective_difference']) < epsilon:
            number_of_null_iterations += 1
    print("{0}% of null iterations".format(number_of_null_iterations / number_of_iterations * 100))

    print("{0} Starting training...".format(step))
    step += 1

    for epoch in range(max_epoch + 1):
        for index, (graph, input_data) in enumerate(inputs_train):
            logits = graph_convolutional_network(graph, input_data)
            logp = F.softmax(logits, dim=0)
            loss = loss_function(logp, labels[train_mask][index])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % 5 == 0:
            # random_logit = torch.tensor([[np.random.rand() for _ in range(output_size)] for _ in degrees])
            # random_loss = loss_function(F.log_softmax(random_logit, 1)[train_mask], labels[train_mask])
            accuracy = evaluate(graph_convolutional_network, inputs_test, labels, train_mask)
            # print(
            #     'Epoch %d, loss %.4f, random loss %.4f, accuracy %.4f' % (
            #         epoch, loss.item(), random_loss.item(), accuracy))
            print('Epoch %d, loss %.4f, accuracy %.4f' % (epoch, loss.item(), accuracy))


if __name__ == '__main__':
    main()
