#!/home/thib/Documents/Travail/CIRRELT/pythonLNS-env/bin/python
import dgl
import torch
import os

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from ALNS.alns_state import CvrpState
from ALNS.generate_instances import generate_cvrp_instance
from NeuralNetwork.create_dataset import retrieve_alns_stats

STATISTICS_DATA_PATH = os.getcwd().rpartition('/')[0] + '/data/'
ALNS_STATISTICS_FILE = '1inst_50nod_40cap_1dep_5000iter_0.8decay_0.35destr_18determ.pickle'

NODE_FEATURES = 3
HIDDEN_DIMENSION = 32
OUTPUT_SIZE = 3
DROPOUT_PROBABILITY = 0.2
MAX_EPOCH = 80
EPSILON = 1e-5

INITIAL_LEARNING_RATE = 0.001
LEARNING_RATE_DECREASE_FACTOR = 0.9

MASK_SEED = 123456


class VanillaGCNLayer(nn.Module):
    def __init__(self, input_node_features, output_feature, in_features_name, out_features_name):
        super(VanillaGCNLayer, self).__init__()

        self.in_features_name = in_features_name
        self.out_features_name = out_features_name

        self.U = torch.empty(output_feature, input_node_features)
        self.V = torch.empty(output_feature, input_node_features)
        nn.init.xavier_normal_(self.U)
        nn.init.xavier_normal_(self.V)

        self.activation = nn.ReLU()

    @staticmethod
    def multiply_weight(features, weight):
        weighted_features = torch.empty(features.shape[0], weight.shape[0])
        for i, feat in enumerate(features):
            weighted_features[i] = torch.mv(weight, feat)

        return weighted_features

    def message_function(self, edges):
        return {self.in_features_name: edges.src[self.in_features_name]}

    def reduce_function(self, nodes):
        # Vanilla ConvNet : h_i^l+1 = ReLU(U^l x h_i^l + V^l x sum h_j^l)
        source_nodes_features = torch.sum(nodes.mailbox[self.in_features_name], dim=1)
        source_nodes_weight = self.multiply_weight(source_nodes_features, self.V)
        destination_node_weight = self.multiply_weight(nodes.data[self.in_features_name], self.U)
        new_features = self.activation(destination_node_weight + source_nodes_weight)

        return {self.out_features_name: new_features}

    def forward(self, graph):
        graph.update_all(message_func=self.message_function, reduce_func=self.reduce_function)

        return graph


class _GatedGCNLayer(nn.Module):
    def __init__(self, input_node_features, output_node_features,
                 input_edge_features, output_edge_features,
                 in_node_features_name='n_feat', out_node_features_name='network_n_feat',
                 in_edge_features_name='e_feat', out_edge_features_name='network_e_feat'):
        super(GatedGCNLayer, self).__init__()

        self.in_node_features_name = in_node_features_name
        self.out_node_features_name = out_node_features_name
        self.in_edge_features_name = in_edge_features_name
        self.out_edge_features_name = out_edge_features_name

        self.U = torch.empty(output_node_features, input_node_features)
        self.V = torch.empty(output_node_features, input_node_features)
        nn.init.xavier_normal_(self.U)
        nn.init.xavier_normal_(self.V)

        self.W1 = torch.empty(output_edge_features, input_edge_features)
        self.W2 = torch.empty(output_edge_features, input_node_features)
        self.W3 = torch.empty(output_edge_features, input_node_features)
        nn.init.xavier_normal_(self.W1)
        nn.init.xavier_normal_(self.W2)
        nn.init.xavier_normal_(self.W3)

        self.activation = nn.ReLU()
        self.h_BN = nn.BatchNorm1d(output_node_features)
        self.e_BN = nn.BatchNorm1d(output_edge_features)
        self.sigmoid = nn.Sigmoid()

    @staticmethod
    def multiply_weight(features, weight):
        weighted_features = torch.empty(features.shape[0], weight.shape[0])
        for i, feat in enumerate(features):
            weighted_features[i] = torch.mv(weight, feat)

        return weighted_features

    @staticmethod
    def hadamard_product(matrix_a, matrix_b):
        if matrix_a.shape != matrix_b.shape:
            raise ArithmeticError
        product = torch.empty(matrix_a.shape)
        for aij, bij, productij in zip(matrix_a.flatten(), matrix_b.flatten(), product.flatten()):
            productij.fill_(aij.item() * bij.item())

        return product

    def message_function(self, edges):
        weighted_neighbor_features = self.multiply_weight(edges.src[self.in_node_features_name], self.V)
        eta = self.sigmoid(edges.data[self.in_edge_features_name]) \
              / (torch.sum(self.sigmoid(edges.data[self.in_edge_features_name]), dim=1) + EPSILON)
        new_neighbor_features = self.hadamard_product(eta, weighted_neighbor_features)
        return {self.out_node_features_name + '_temp': new_neighbor_features}

    def reduce_function(self, nodes):
        # GatedGCN : h_i^l+1 = h_i^l + ReLU(BN(U^l x h_i^l + sum eta_ij^l * (V^l x h_j^l) ))
        # See https://arxiv.org/abs/1711.07553 and https://arxiv.org/abs/1906.01227
        source_nodes_weight = torch.sum(nodes.mailbox[self.out_node_features_name + '_temp'], dim=1)
        destination_node_weight = self.multiply_weight(nodes.data[self.in_node_features_name], self.U)
        new_features = self.activation(self.h_BN(destination_node_weight + source_nodes_weight))

        return {self.out_node_features_name + '_temp': new_features}

    def update_edge(self, edges):
        new_edge_features = edges.data[self.in_edge_features_name] \
                            + self.activation(self.e_BN())

        return {self.out_edge_features_name: new_edge_features}

    def forward(self, graph):
        graph.update_all(message_func=self.message_function, reduce_func=self.reduce_function)
        graph.apply_edges(func=self.update_edge)
        graph.ndata[self.out_node_features_name] = graph.ndata.pop(self.out_node_features_name + '_temp')

        return graph


class GatedGCNLayer(nn.Module):
    def __init__(self, input_node_features, output_node_features,
                 input_edge_features, output_edge_features,
                 dropout_probability):
        super().__init__()

        self.input_node_features = input_node_features
        self.output_node_features = output_node_features
        self.input_edge_features = input_edge_features
        self.output_edge_features = output_edge_features

        # This embeddings are used to change the dimension of the node and edge features from one layer to another
        # Otherwise it would be impossible to add the previous feature vector to the newly computed one
        self.embedding_node = nn.Linear(input_node_features, output_node_features, bias=False)
        self.embedding_edge = nn.Linear(input_edge_features, output_edge_features, bias=False)

        # This embedding is used to be able to add the edge feature vector to the node feature vector
        self.embedding_eta = nn.Linear(output_edge_features, output_node_features, bias=False)

        self.dropout_probability = dropout_probability

        self.U = nn.Linear(input_node_features, output_node_features, bias=True)
        self.V = nn.Linear(input_node_features, output_node_features, bias=True)

        self.W1 = nn.Linear(input_edge_features, output_edge_features, bias=True)
        self.W2 = nn.Linear(input_node_features, output_edge_features, bias=True)
        self.W3 = nn.Linear(input_node_features, output_edge_features, bias=True)

        self.activation = nn.ReLU()
        self.h_BN = nn.BatchNorm1d(output_node_features)
        self.e_BN = nn.BatchNorm1d(output_edge_features)
        self.sigmoid = nn.Sigmoid()

    def message_function(self, edges):
        Vh_j = edges.src['Vh']
        e_ij = edges.data['e'] + self.activation(self.e_BN(edges.data['W1e'] + edges.src['W2h'] + edges.dst['W3h']))
        edges.data['e'] = e_ij

        return {'Vh_j': Vh_j, 'e_ij': e_ij}

    def reduce_function(self, nodes):
        Uh_i = nodes.data['Uh']
        Vh_j = nodes.mailbox['Vh_j']

        e = nodes.mailbox['e_ij']
        sigma_ij = self.embedding_eta(self.sigmoid(e))

        h = nodes.data['h'] + self.activation(self.h_BN(Uh_i + torch.sum(sigma_ij * Vh_j, dim=1)
                                                        / (torch.sum(sigma_ij, dim=1) + EPSILON)))

        return {'h': h}

    def forward(self, graph, h, e):
        graph.ndata['h'] = self.embedding_node(h)
        graph.ndata['Uh'] = self.U(h)
        graph.ndata['Vh'] = self.V(h)
        graph.ndata['W2h'] = self.W2(h)
        graph.ndata['W3h'] = self.W3(h)

        graph.edata['e'] = self.embedding_edge(e)
        graph.edata['W1e'] = self.W1(e)

        graph.update_all(message_func=self.message_function, reduce_func=self.reduce_function)
        h = graph.ndata['h']
        e = graph.edata['e']

        h = torch.nn.functional.dropout(h, self.dropout_probability)
        e = torch.nn.functional.dropout(e, self.dropout_probability)

        return h, e


class GCN(nn.Module):
    def __init__(self, input_node_features, hidden_node_dimension,
                 input_edge_features, hidden_edge_dimension,
                 output_feature, dropout_probability):
        super(GCN, self).__init__()
        self.convolution = GatedGCNLayer(input_node_features, hidden_node_dimension,
                                         input_edge_features, hidden_edge_dimension,
                                         dropout_probability)
        self.convolution2 = GatedGCNLayer(hidden_node_dimension, hidden_node_dimension,
                                          hidden_edge_dimension, hidden_edge_dimension,
                                          dropout_probability)
        self.convolution3 = GatedGCNLayer(hidden_node_dimension, hidden_node_dimension,
                                          hidden_edge_dimension, hidden_edge_dimension,
                                          dropout_probability)
        self.convolution4 = GatedGCNLayer(hidden_node_dimension, hidden_node_dimension,
                                          hidden_edge_dimension, hidden_edge_dimension,
                                          dropout_probability)
        self.linear = nn.Linear(hidden_node_dimension, output_feature)

    def forward(self, graph, h, e):
        h, e = self.convolution(graph, h, e)
        h, e = self.convolution2(graph, h, e)
        h, e = self.convolution3(graph, h, e)
        h, e = self.convolution4(graph, h, e)

        # Return a tensor of shape (hidden_dimension)
        h = torch.mean(h, dim=0)
        h = self.linear(h)
        return h


def evaluate(network, inputs_test, labels, train_mask):
    test_mask = ~train_mask
    network.eval()
    with torch.no_grad():
        correct = 0
        for index, graph in enumerate(inputs_test):
            logits = network(graph, graph.ndata['n_feat'], graph.edata['e_feat'])
            logp = F.softmax(logits, dim=0)
            # torch.max -> (max, argmax), so we only keep the argmax
            predicted_class = torch.argmax(logp, dim=0).item()
            true_class = torch.argmax(labels[test_mask][index], dim=0).item()
            correct += predicted_class == true_class
        return correct / len(test_mask)


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
    dgl_graph = dgl.DGLGraph()
    dgl_graph.from_networkx(nx_graph=nx_graph)
    dgl_graph.set_n_initializer(dgl.init.zero_initializer)

    return dgl_graph


def generate_graph_node_features(graph, graph_index, cvrp_state, alns_instance_statistics):
    nx_graph = cvrp_state.instance
    number_of_nodes = nx_graph.number_of_nodes()

    node_features = [[cvrp_state.capacity - nx_graph.nodes[node]['demand'],
                      1 if nx_graph.nodes[node]['isDepot'] else 0,
                      1 if node in alns_instance_statistics['Statistics'][graph_index]['destroyed_nodes'] else 0]
                     for node in range(number_of_nodes)]
    edge_features = [[nx_graph.edges[u, v]['weight'],
                      1 if (u, v) in alns_instance_statistics['Statistics'][graph_index]['list_of_edges'] else 0]
                     for u in range(number_of_nodes) for v in range(number_of_nodes) if u != v]

    node_features_tensor = torch.tensor(node_features, dtype=torch.float)
    edge_features_tensor = torch.tensor(edge_features, dtype=torch.float)
    graph.ndata['n_feat'] = node_features_tensor
    graph.edata['e_feat'] = edge_features_tensor

    return graph


def generate_input_graphs_from_cvrp_state(cvrp_state, alns_instance_statistics):
    nx_graph = cvrp_state.instance
    list_of_dgl_graphs = \
        [generate_cvrp_graph(nx_graph) for _ in range(len(alns_instance_statistics['Statistics']))]
    # All graphs must have the same number of node
    # They actually represent the same CVRP problem with different solution
    # The data is the CVRP state at different iterations of the ALNS heuristic

    for i, graph in enumerate(list_of_dgl_graphs):
        generate_graph_node_features(graph, i, cvrp_state, alns_instance_statistics)

    inputs = list_of_dgl_graphs
    inputs_train = []
    inputs_test = []
    train_mask = []
    np.random.seed(MASK_SEED)
    for index, single_input in enumerate(inputs):
        if np.random.randint(0, 4) > 0:
            inputs_train.append(single_input)
            train_mask.append(1)
        else:
            inputs_test.append(single_input)
            train_mask.append(0)

    train_mask = torch.tensor(train_mask).bool()

    return inputs_train, inputs_test, train_mask


def generate_labels_from_cvrp_state(alns_instance_statistics, epsilon=EPSILON):
    labels = torch.tensor([[1, 0, 0] if iteration['objective_difference'] > 0
                           else [0, 1, 0] if abs(iteration['objective_difference']) <= epsilon else [0, 0, 1]
                           for iteration in alns_instance_statistics['Statistics']],
                          dtype=torch.float)

    return labels


def main(alns_statistics_file=ALNS_STATISTICS_FILE, hidden_dimension=HIDDEN_DIMENSION, output_size=OUTPUT_SIZE,
         dropout_probability=DROPOUT_PROBABILITY,
         max_epoch=MAX_EPOCH, epsilon=EPSILON,
         initial_learning_rate=INITIAL_LEARNING_RATE, learning_rate_decrease_factor=LEARNING_RATE_DECREASE_FACTOR):
    step = 1

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

    inputs_train, inputs_test, train_mask = generate_input_graphs_from_cvrp_state(cvrp_state, alns_instance_statistics)
    labels = generate_labels_from_cvrp_state(alns_instance_statistics, epsilon)
    number_of_node_features = len(inputs_test[0].ndata['n_feat'][0])
    number_of_edge_features = len(inputs_test[0].edata['e_feat'][0])
    print("{0} Created inputs and labels".format(step))
    step += 1

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    graph_convolutional_network = GCN(input_node_features=number_of_node_features,
                                      hidden_node_dimension=hidden_dimension,
                                      input_edge_features=number_of_edge_features,
                                      hidden_edge_dimension=hidden_dimension,
                                      output_feature=output_size,
                                      dropout_probability=dropout_probability)
    graph_convolutional_network = graph_convolutional_network.to(device)
    print("{0} Created GCN".format(step))
    step += 1

    optimizer = torch.optim.Adam(graph_convolutional_network.parameters(), lr=initial_learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=learning_rate_decrease_factor)
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
        loss = torch.tensor([1], dtype=torch.float)
        for index, graph in enumerate(inputs_train):
            logits = graph_convolutional_network(graph, graph.ndata['n_feat'], graph.edata['e_feat'])
            logp = F.softmax(logits, dim=0)
            loss = loss_function(logp, labels[train_mask][index])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step(loss)

        if epoch % 5 == 0:
            accuracy = evaluate(graph_convolutional_network, inputs_test, labels, train_mask)
            print('Epoch %d, loss %.6f, accuracy %.4f' % (epoch, loss.item(), accuracy))


if __name__ == '__main__':
    main()
