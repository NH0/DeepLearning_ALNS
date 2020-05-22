import dgl
import torch
import os
import pickle

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from ALNS.alns_state import CvrpState
from ALNS.generate_instances import generate_cvrp_instance
from NeuralNetwork.create_dataset import retrieve_alns_stats

DATASET_PREFIX = 'inputs_mask_labels_'
STATISTICS_DATA_PATH = os.getcwd().rpartition('/')[0] + '/data/'
ALNS_STATISTICS_FILE = 'dataset_50-50_1inst_50nod_40cap_1dep_50000iter_0.8decay_0.35destr_18determ.pickle'
DATASET_PATH = STATISTICS_DATA_PATH + DATASET_PREFIX + ALNS_STATISTICS_FILE
MODEL_PARAMETERS_PATH = STATISTICS_DATA_PATH + 'parametersGCN'

HIDDEN_NODE_DIMENSIONS = [64, 32, 16, 8]
HIDDEN_EDGE_DIMENSIONS = [32, 16, 16, 8]
HIDDEN_LINEAR_DIMENSION = 32
OUTPUT_SIZE = 3
DROPOUT_PROBABILITY = 0.2
MAX_EPOCH = 5000
EPSILON = 1e-5

INITIAL_LEARNING_RATE = 0.00001
LEARNING_RATE_DECREASE_FACTOR = 0.9

MASK_SEED = 123456


class GatedGCNLayer(nn.Module):
    """
    Defines a layer of a Gated Graph Convolutional Network based on https://arxiv.org/pdf/1711.07553v2.pdf.

    The equation defining the message passing is the following :
    h = hi + ReLU(BN(U x hi + Sum(eta_ij * V x hj)))
    where :
    eta_ij = sigmoid(e_ij) / (Sum(sigmoid(e_ij)) + epsilon)
    and :
    e_ij = e_ij + ReLU(BN(W1 x e_ij + W2 x hi + W3 x hj))

    BN is a Batch normalization.

    In order to have different feature sizes between layers, we add a linear transformation
    for nodes to hi :
    h = embedding_node x hi + ReLU(BN(W1 x hi + Sum(eta_ij * W2 x hj)))
    for edges to e_ij :
    e_ij = embedding_edge x e_ij + ReLU(BN(W1 x e_ij + W2 x hi + W3 x hj))


    And to allow different sizes between node features and edge features, we also add a linear transformation to eta_ij
    h = embedding_node x hi + ReLU(BN(W1 x hi + Sum(embedding_eta x eta_ij * W2 x hj)))
    """

    def __init__(self, input_node_features, output_node_features,
                 input_edge_features, output_edge_features,
                 dropout_probability, has_dropout=False):
        super(GatedGCNLayer, self).__init__()

        self.input_node_features = input_node_features
        self.output_node_features = output_node_features
        self.input_edge_features = input_edge_features
        self.output_edge_features = output_edge_features

        # This embeddings are used to change the dimension of the node and edge features from one layer to another
        # Otherwise each layer would have the same number of features
        self.embedding_node = nn.Linear(input_node_features, output_node_features, bias=False)
        self.embedding_edge = nn.Linear(input_edge_features, output_edge_features, bias=False)

        # This embedding is used to be able to add the edge feature vector to the node feature vector
        self.embedding_eta = nn.Linear(output_edge_features, output_node_features, bias=False)

        self.has_dropout = has_dropout
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

        if self.has_dropout:
            h = torch.nn.functional.dropout(h, self.dropout_probability)
            e = torch.nn.functional.dropout(e, self.dropout_probability)

        return h, e


class GCN(nn.Module):
    """
    Classifies an alns iteration for the CVRP (destruction & reconstruction).

    The network predicts whether the iteration will improve, worsen or keep the total cost of the CVRP solution.
    The network is based on Gated Graph Convolution layers followed by a Fully connected layer with and output of size
    3 (for the 3 possible classes).
    """

    def __init__(self,
                 input_node_features, hidden_node_dimension_list,
                 input_edge_features, hidden_edge_dimension_list,
                 hidden_linear_dimension,
                 output_feature,
                 dropout_probability,
                 device):
        super(GCN, self).__init__()

        if len(hidden_node_dimension_list) != len(hidden_edge_dimension_list):
            print("Node dimensions and edge dimensions lists aren't the same size !\nExiting...")
            exit(1)

        self.convolutions = [GatedGCNLayer(input_node_features, hidden_node_dimension_list[0],
                                           input_edge_features, hidden_edge_dimension_list[0],
                                           dropout_probability).to(device)]
        self.add_module('convolution1', self.convolutions[0])

        for i in range(1, len(hidden_node_dimension_list)):
            self.convolutions.append(GatedGCNLayer(hidden_node_dimension_list[i - 1], hidden_node_dimension_list[i],
                                                   hidden_edge_dimension_list[i - 1], hidden_edge_dimension_list[i],
                                                   dropout_probability).to(device))
            self.add_module('convolution' + str(i + 1), self.convolutions[-1])

        self.linear1 = nn.Linear(hidden_node_dimension_list[-1], hidden_linear_dimension)
        self.linear2 = nn.Linear(hidden_linear_dimension, output_feature)

    def forward(self, graph, h, e):
        for convolution in self.convolutions:
            h, e = convolution(graph, h, e)

        # Return a tensor of shape (hidden_dimension)
        h = torch.mean(h, dim=0)
        h = self.linear1(h)
        h = self.linear2(h)
        return h


def evaluate(network, inputs_test, labels, train_mask):
    """
    Evaluate a neural network on a given test set.

    Parameters
    ----------
    network : the network to evaluate
    inputs_test : the test dataset, containing DGL graphs
    labels : the expected values to be returned by the network
    train_mask : the inverse of mask to apply on the labels to keep only the labels corresponding to the test set

    Returns
    -------
    The proportion of right predictions
    """
    # Inverse the mask to have the test mask
    test_mask = ~train_mask
    network.eval()
    with torch.no_grad():
        correct = 0
        for index, graph in enumerate(inputs_test):
            logits = network(graph, graph.ndata['n_feat'], graph.edata['e_feat'])
            logp = F.softmax(logits, dim=0)
            predicted_class = torch.argmax(logp, dim=0).item()
            true_class = torch.argmax(labels[test_mask][index], dim=0).item()
            correct += predicted_class == true_class

    return correct / len(inputs_test)


def evaluate_random(labels, train_mask, number_of_test_values):
    test_mask = ~train_mask
    correct = 0
    for i in range(number_of_test_values):
        true_class = torch.argmax(labels[test_mask][i], dim=0).item()
        correct += np.random.randint(0, 3) == true_class

    return correct / number_of_test_values


def make_complete_graph(initial_state):
    """
    Make the instance of a CVRP state complete (each node connected to every other node).
    Nodes aren't connected to themselves.
    The modification is in place.

    Necessary in order to have the distance information included in the graph (weight of each edge).

    Parameters
    ----------
    initial_state : a CVRP state containing all the characteristics of the CVRP problem. Called initial because it is
                    obtained when creating a CVRP state.

    Returns
    -------
    None
    """
    number_of_nodes = initial_state.instance.number_of_nodes()
    # Create a list containing all possible edges
    edges_in_complete_graph = [(u, v) for u in range(number_of_nodes) for v in range(number_of_nodes) if u != v]
    for u, v in edges_in_complete_graph:
        # Networkx will not add the edge if it already exists
        initial_state.instance.add_edge(u, v, weight=initial_state.distances[u][v])


def create_cvrp_state(size, number_of_depots, capacity, seed):
    """
    Create a CVRP state with given parameters.

    Parameters
    ----------
    size : the number of nodes (depots and clients)
    number_of_depots : the number of depots, for the moment consider it equal to 1
    capacity : capacity of the delivery vehicle
    seed : the seed used to generate the instance

    Returns
    -------
    A CVRP state as defined in ALNS.alns_state
    """
    cvrp_instance = generate_cvrp_instance(size=size, capacity=capacity, number_of_depots=number_of_depots, seed=seed)
    # Create an empty state
    initial_state = CvrpState(cvrp_instance, size=size, capacity=capacity, number_of_depots=number_of_depots,
                              collect_alns_statistics=False, seed=seed)
    # initial_solution = generate_initial_solution(initial_state)
    make_complete_graph(initial_state)

    return initial_state


def generate_cvrp_graph(nx_graph):
    """
    Convert a networkx graph to a DGL graph.

    Parameters
    ----------
    nx_graph : a networkx graph

    Returns
    -------
    dgl_graph the nx_graph converted to DGL
    """
    dgl_graph = dgl.DGLGraph()
    dgl_graph.from_networkx(nx_graph=nx_graph)
    dgl_graph.set_n_initializer(dgl.init.zero_initializer)

    return dgl_graph


def generate_graph_features_from_statistics(graph, graph_index, cvrp_state, alns_instance_statistics, device):
    """
    Add node and edge features to a DGL graph representing a CVRP instance.

    The node features are :
    -> capacity - demand : (float) where capacity is the capacity of the delivery vehicle and demand is the demand of
                            the node
    -> isDepot : (boolean 0 or 1) 1 is the node is a depot
    -> isDestroyed : (boolean 0 or 1) 1 if the node is part of the destroyed node during the current ALNS iteration

    The edge features are :
    -> weight : (float) the distance between two nodes (= the cost of using this edge in a CVRP solution)
    -> isUsed : (boolean 0 or 1) 1 if the edge is used in the current CVRP solution

    Parameters
    ----------
    graph : the DGL graph, representing an iteration during the ALNS execution. It doesn't contain any information yet.
            It is simply the complete graph corresponding to the CVRP instance.
    graph_index : the index of the graph, corresponding to the index of the iteration in the statistics. The destroyed
                    nodes and the edges in the solution depend on this index as the solution evolves during the
                    consecutive iterations.
    cvrp_state : the cvrp state corresponding to the CVRP problem. It contains all the information on the CVRP problem
                 currently studied.
    alns_instance_statistics : the statistics saved during the execution of the ALNS algorithm. It contains the
                               destroyed nodes, the edges of the solution and the difference in the total cost (not used
                               in this function).
    device:  CPU or CUDA depending on the device used for execution

    Returns
    -------
    None
    """
    nx_graph = cvrp_state.instance
    number_of_nodes = nx_graph.number_of_nodes()

    node_features = [[cvrp_state.capacity - nx_graph.nodes[node]['demand'],
                      1 if nx_graph.nodes[node]['isDepot'] else 0,
                      1 if node in alns_instance_statistics['Statistics'][graph_index]['destroyed_nodes'] else 0]
                     for node in range(number_of_nodes)]
    edge_features = [[nx_graph.edges[u, v]['weight'],
                      1 if (u, v) in alns_instance_statistics['Statistics'][graph_index]['list_of_edges'] else 0]
                     for u in range(number_of_nodes) for v in range(number_of_nodes) if u != v]

    node_features_tensor = torch.tensor(node_features, dtype=torch.float, device=device)
    edge_features_tensor = torch.tensor(edge_features, dtype=torch.float, device=device)
    graph.ndata['n_feat'] = node_features_tensor
    graph.edata['e_feat'] = edge_features_tensor


def generate_train_and_test_sets_from_cvrp_state(cvrp_state, alns_instance_statistics, device):
    """
    Create the train and test sets.

    First it generates the graphs, then it adds the features (node and edge) to each graphs and finally generates the
    train mask for separating the train and test sets.

    Parameters
    ----------
    cvrp_state : the cvrp state corresponding to the CVRP problem. It contains all the information on the CVRP problem
                 currently studied.
    alns_instance_statistics : the statistics saved during the execution of the ALNS algorithm. It contains the
                               destroyed nodes, the edges of the solution and the difference in the total cost (not used
                               in this function).
    device : CPU or CUDA depending on the device used for execution

    Returns
    -------
    train_set, test_set, train_mask
    """
    nx_graph = cvrp_state.instance
    list_of_dgl_graphs = \
        [generate_cvrp_graph(nx_graph) for _ in range(len(alns_instance_statistics['Statistics']))]
    # All graphs must have the same number of node
    # They actually represent the same CVRP problem with different solution
    # The data is the CVRP state at different iterations of the ALNS heuristic

    for i, graph in enumerate(list_of_dgl_graphs):
        generate_graph_features_from_statistics(graph, i, cvrp_state, alns_instance_statistics, device)

    inputs = list_of_dgl_graphs
    train_set = []
    test_set = []
    train_mask = []
    np.random.seed(MASK_SEED)
    for index, single_input in enumerate(inputs):
        if np.random.randint(0, 4) > 0:
            train_set.append(single_input)
            train_mask.append(1)
        else:
            test_set.append(single_input)
            train_mask.append(0)

    train_mask = torch.tensor(train_mask).bool()

    return train_set, test_set, train_mask


def generate_labels_from_cvrp_state(alns_instance_statistics, device, epsilon=EPSILON):
    """
    Generate the labels for each ALNS iteration. The labels can be one in 3 values :
    - (1,0,0) : if the iteration worsens the current cost
    - (0,1,0) : if the iteration doesn't change the current cost
    - (0,0,1) : if the iteration improves the current cost

    Parameters
    ----------
    alns_instance_statistics : the statistics saved during the execution of the ALNS algorithm. It contains the
                               destroyed nodes, the edges of the solution and the difference in the total
                               cost (objective_difference).
    device : CPU or CUDA depending on the device used for execution
    epsilon : small value to avoid comparing floats to 0.0

    Returns
    -------
    labels
    """
    labels = torch.tensor([[1, 0, 0] if iteration['objective_difference'] > 0
                           else [0, 1, 0] if abs(iteration['objective_difference']) <= epsilon
                           else [0, 0, 1]
                           for iteration in alns_instance_statistics['Statistics']],
                          dtype=torch.float, device=device)

    return labels


def create_dataset_from_statistics(alns_statistics_file,
                                   device,
                                   epsilon=EPSILON):
    step = 1
    """
    Retrieve the statistics saved during the ALNS execution.
    """
    alns_statistics_path = STATISTICS_DATA_PATH + alns_statistics_file
    # Warning : can be a list of dictionaries, here considered to be a single dictionary
    alns_instance_statistics = retrieve_alns_stats(alns_statistics_path)
    if type(alns_instance_statistics) != dict:
        print("Error, the stats file contains different CVRP instances.")
        return -1
    print("\t{} Retrieved alns statistics".format(step))
    step += 1

    """
    Create the CVRP state using the given parameters.
    """
    cvrp_state = create_cvrp_state(size=alns_instance_statistics['Size'],
                                   number_of_depots=alns_instance_statistics['Number_of_depots'],
                                   capacity=alns_instance_statistics['Capacity'],
                                   seed=alns_instance_statistics['Seed'])
    print("\t{} Created new cvrp state".format(step))
    step += 1
    print("\t{} Creating inputs and labels ... ".format(step), end='', flush=True)
    inputs_train, inputs_test, train_mask = generate_train_and_test_sets_from_cvrp_state(cvrp_state,
                                                                                         alns_instance_statistics,
                                                                                         device)
    print("created inputs, ", end='', flush=True)
    labels = generate_labels_from_cvrp_state(alns_instance_statistics, device, epsilon)

    print("and created labels", flush=True)

    return inputs_train, inputs_test, train_mask, labels


def pickle_dataset(filename, inputs_train, inputs_test, train_mask, labels):
    with open(filename, 'wb') as dataset_file:
        try:
            pickle.dump({'inputs_training': inputs_train,
                         'inputs_testing': inputs_test,
                         'train_mask': train_mask,
                         'labels': labels}, dataset_file)
        except pickle.PicklingError:
            print("Unable to pickle data...\nExiting now.")
            exit(1)
        print("Successfully saved the data in {}".format(filename))


def unpickle_dataset(dataset_path):
    with open(dataset_path, 'rb') as dataset_file:
        try:
            dataset = pickle.load(dataset_file)
        except pickle.UnpicklingError:
            print("Error, couldn't unpickle the dataset.\nExiting now.")
            exit(2)

    return dataset['inputs_training'], dataset['inputs_testing'], dataset['train_mask'], dataset['labels']


def display_proportion_of_null_iterations(train_mask, labels, training_set_size, device):
    number_of_iterations = len(train_mask)
    number_of_total_null_iterations = 0
    number_of_train_null_iterations = 0
    null_label = torch.tensor([0, 1, 0], dtype=torch.float, device=device)
    for index, iteration in enumerate(labels):
        if torch.equal(iteration, null_label):
            number_of_total_null_iterations += 1
            if train_mask[index] == 1:
                number_of_train_null_iterations += 1
    print("{:.2%} of total null iterations".format(
        round(number_of_total_null_iterations / number_of_iterations, 4)
    ))
    print("{:.2%} of null iterations in training set".format(
        round(number_of_train_null_iterations / training_set_size, 4)
    ))
    print("Dataset size : {}".format(number_of_iterations))
    print("Training set size : {}".format(training_set_size))


def save_model_parameters(GCN_model,
                          hidden_node_dimensions, hidden_edge_dimensions, hidden_linear_dimension,
                          initial_learning_rate,
                          epoch,
                          device):
    name_model_parameters_file = '_ep' + str(epoch) + '_ndim'
    for dim in hidden_node_dimensions:
        name_model_parameters_file += str(dim) + '.'
    name_model_parameters_file += '_edim'
    for dim in hidden_edge_dimensions:
        name_model_parameters_file += str(dim) + '.'
    name_model_parameters_file += '_lindim' + str(hidden_linear_dimension)
    name_model_parameters_file += '_lr' + str(initial_learning_rate)
    name_model_parameters_file += '_dev' + device
    name_model_parameters_file += '.pt'
    torch.save(GCN_model.state_dict(), MODEL_PARAMETERS_PATH + name_model_parameters_file)
    print("Successfully saved the model's parameters in {}".format(MODEL_PARAMETERS_PATH + name_model_parameters_file))


def main(recreate_dataset=False,
         hidden_node_dimensions=None,
         hidden_edge_dimensions=None,
         hidden_linear_dimension=HIDDEN_LINEAR_DIMENSION,
         output_size=OUTPUT_SIZE,
         dropout_probability=DROPOUT_PROBABILITY,
         max_epoch=MAX_EPOCH, epsilon=EPSILON,
         initial_learning_rate=INITIAL_LEARNING_RATE,
         learning_rate_decrease_factor=LEARNING_RATE_DECREASE_FACTOR,
         save_parameters_on_exit=True,
         load_parameters_from_file=None,
         **keywords_args):
    # Avoid mutable default arguments
    if hidden_edge_dimensions is None:
        hidden_edge_dimensions = HIDDEN_EDGE_DIMENSIONS
    if hidden_node_dimensions is None:
        hidden_node_dimensions = HIDDEN_NODE_DIMENSIONS

    """
    Use GPU if available.
    """
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    print("#" * 50)
    print("# Hidden node dimensions : {}".format(hidden_node_dimensions))
    print("# Hidden edge dimensions : {}".format(hidden_edge_dimensions))
    print("# Hidden linear dimension : {}".format(hidden_linear_dimension))
    print("# Dropout probability : {}".format(dropout_probability))
    print("# Max epoch : {}".format(max_epoch))
    print("# Initial learning rate : {}".format(initial_learning_rate))
    print("# Device : {}".format(device))
    print("#" * 50)

    if recreate_dataset:
        print("Creating dataset from ALNS statistics :")
        if 'alns_statistics_file' not in keywords_args:
            alns_statistics_file = ALNS_STATISTICS_FILE
        else:
            alns_statistics_file = keywords_args['alns_statistics_file']
        """
        Create the train and test sets.
        """
        inputs_train, inputs_test, train_mask, labels = create_dataset_from_statistics(alns_statistics_file,
                                                                                       device,
                                                                                       epsilon)
        print("Created dataset !")
        if 'pickle_dataset' in keywords_args:
            if keywords_args['pickle_dataset']:
                dataset_filename = DATASET_PREFIX + alns_statistics_file
                pickle_dataset(dataset_filename, inputs_train, inputs_test, train_mask, labels)
    else:
        print("Retrieving dataset ... ", end='', flush=True)
        if 'dataset_path' not in keywords_args:
            dataset_path = DATASET_PATH
        else:
            dataset_path = keywords_args['dataset_path']
        inputs_train, inputs_test, train_mask, labels = unpickle_dataset(dataset_path)
        print("Done !", flush=True)

    number_of_node_features = len(inputs_test[0].ndata['n_feat'][0])
    number_of_edge_features = len(inputs_test[0].edata['e_feat'][0])

    """
    Create the gated graph convolutional network
    """
    graph_convolutional_network = GCN(input_node_features=number_of_node_features,
                                      hidden_node_dimension_list=hidden_node_dimensions,
                                      input_edge_features=number_of_edge_features,
                                      hidden_edge_dimension_list=hidden_edge_dimensions,
                                      hidden_linear_dimension=hidden_linear_dimension,
                                      output_feature=output_size,
                                      dropout_probability=dropout_probability,
                                      device=device)
    graph_convolutional_network = graph_convolutional_network.to(device)
    print("Created GCN", flush=True)
    if load_parameters_from_file is not None:
        try:
            graph_convolutional_network.load_state_dict(torch.load(load_parameters_from_file))
            graph_convolutional_network.eval()
            print("Loaded parameters values from {}".format(load_parameters_from_file))
        except (pickle.UnpicklingError, TypeError, RuntimeError) as exception_value:
            print("Unable to load parameters from {}".format(load_parameters_from_file))
            print("Exception : {}".format(exception_value))
            should_continue = ''
            while should_continue != 'y' or should_continue != 'n':
                should_continue = input("Continue anyway with random parameters ? (y/n) ")
            if should_continue == 'n':
                exit(1)

    """
    Define the optimizer, the learning rate scheduler and the loss function.
    We use the Adam optimizer and a MSE loss.
    """
    optimizer = torch.optim.Adam(graph_convolutional_network.parameters(), lr=initial_learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=learning_rate_decrease_factor)
    loss_function = nn.MSELoss()

    """
    Display the proportion of null iterations (iterations that do not change the cost value of the CVRP solution.
    """
    display_proportion_of_null_iterations(train_mask, labels, len(inputs_train), device)

    print("\nStarting training...\n")

    """
    Train the network.
    """
    for epoch in range(max_epoch + 1):
        try:
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
                random_accuracy = evaluate_random(labels, train_mask, len(inputs_test))
                print("Epoch {:d}, loss {:f.6}, accuracy {:f.4}, random accuracy {:f.4}"
                      .format(epoch, loss.item(), accuracy, random_accuracy))
        except KeyboardInterrupt:
            if save_parameters_on_exit:
                print("Saving parameters before quiting ...", flush=True)
                save_model_parameters(graph_convolutional_network,
                                      hidden_node_dimensions, hidden_edge_dimensions, hidden_linear_dimension,
                                      initial_learning_rate, epoch, device)
            exit(0)

    if save_parameters_on_exit:
        save_model_parameters(graph_convolutional_network,
                              hidden_node_dimensions, hidden_edge_dimensions, hidden_linear_dimension,
                              initial_learning_rate, max_epoch, device)


if __name__ == '__main__':
    main()
