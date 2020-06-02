import dgl
import torch
import pickle

import src.NeuralNetwork.parameters as parameters

from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from src.ALNS.CVRP.CVRP import CvrpState
from src.ALNS.CVRP.generate_cvrp_graph import generate_cvrp_instance
from src.NeuralNetwork.Dataset.retrieve_alns_stats import retrieve_alns_stats
from src.NeuralNetwork.Dataset.dataset import CVRPDataSet

STATISTICS_DATA_PATH = parameters.STATISTICS_DATA_PATH
ALNS_STATISTICS_FILE = parameters.ALNS_STATISTICS_FILE
DATASET_PATH = parameters.DATASET_PATH
MODEL_PARAMETERS_PATH = parameters.MODEL_PARAMETERS_PATH

EPSILON = parameters.EPSILON

MASK_SEED = parameters.MASK_SEED
BATCH_SIZE = parameters.BATCH_SIZE


def collate(sample):
    graphs, labels = map(list, zip(*sample))
    graph_batch = dgl.batch(graphs)
    return graph_batch, torch.tensor(labels, device=labels[0].device)


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


def generate_inputs_from_cvrp_state(cvrp_state, alns_instance_statistics, device):
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

    return inputs


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
    labels = torch.tensor([0 if iteration['objective_difference'] > 0
                           else 1 if abs(iteration['objective_difference']) <= epsilon
                           else 2
                           for iteration in alns_instance_statistics['Statistics']],
                          dtype=torch.long, device=device)

    return labels


def create_dataset_from_statistics(alns_statistics_file,
                                   device,
                                   batch_size=BATCH_SIZE,
                                   epsilon=EPSILON):
    step = 1
    """
    Retrieve the statistics saved during the ALNS execution.
    """
    alns_statistics_path = STATISTICS_DATA_PATH + alns_statistics_file
    # Warning : can be a list of dictionaries, here considered to be a single dictionary
    alns_instance_statistics = retrieve_alns_stats(alns_statistics_path)
    if len(alns_instance_statistics) != 1:
        print("Error, the stats file contains different CVRP instances.\nUsing only first instance.")
    alns_instance_statistics = alns_instance_statistics[0]
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
    inputs = generate_inputs_from_cvrp_state(cvrp_state, alns_instance_statistics, device)
    print("created inputs, ", end='', flush=True)
    labels = generate_labels_from_cvrp_state(alns_instance_statistics, device, epsilon)
    print("and created labels", flush=True)

    dataset = CVRPDataSet(inputs, labels)
    dataset_size = len(dataset)
    train_size = int(0.8 * dataset_size)
    torch.manual_seed(MASK_SEED)
    train_set, test_set = random_split(dataset, [train_size, dataset_size - train_size])
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, collate_fn=collate)
    test_loader = DataLoader(dataset=test_set, batch_size=1, collate_fn=collate)

    return train_loader, test_loader


def pickle_dataset(dataset_name, train_loader, test_loader):
    with open(DATASET_PATH + dataset_name, 'wb') as dataset_file:
        try:
            pickle.dump({'train_loader': train_loader,
                         'test_loader': test_loader}, dataset_file)
        except pickle.PicklingError:
            print("Unable to pickle data...\nExiting now.")
            exit(1)
        print("Successfully saved the data in {}".format(DATASET_PATH + dataset_name))


def unpickle_dataset(dataset_name):
    with open(DATASET_PATH + dataset_name, 'rb') as dataset_file:
        try:
            dataset = pickle.load(dataset_file)
        except pickle.UnpicklingError:
            print("Error, couldn't unpickle the dataset.\nExiting now.")
            exit(2)

    return dataset['train_loader'], dataset['test_loader']
