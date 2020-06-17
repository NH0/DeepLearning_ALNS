import dgl
import torch

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

DEVICE = parameters.DEVICE


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
        [generate_cvrp_graph(nx_graph) for _ in alns_instance_statistics['Statistics']]
    # All graphs must have the same number of node
    # They actually represent the same CVRP problem with different solution
    # The data is the CVRP state at different iterations of the ALNS heuristic

    for i, graph in enumerate(list_of_dgl_graphs):
        generate_graph_features_from_statistics(graph, i, cvrp_state, alns_instance_statistics, device)

    inputs = list_of_dgl_graphs

    return inputs


def generate_labels_from_cvrp_state(alns_instance_statistics, device=DEVICE, epsilon=EPSILON):
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


def generate_inputs_and_labels_for_single_instance(single_instance_statistics, device=DEVICE, epsilon=EPSILON):
    step = 1
    print("\t{} Retrieved one instance's statistics".format(step))
    step += 1

    """
    Create the CVRP state using the given parameters.
    """
    cvrp_state = create_cvrp_state(size=single_instance_statistics['Size'],
                                   number_of_depots=single_instance_statistics['Number_of_depots'],
                                   capacity=single_instance_statistics['Capacity'],
                                   seed=single_instance_statistics['Seed'])
    print("\t{} Created new cvrp state".format(step))
    step += 1
    print("\t{} Creating inputs and labels ... ".format(step), end='', flush=True)
    inputs = generate_inputs_from_cvrp_state(cvrp_state, single_instance_statistics, device)
    print("created inputs, ", end='', flush=True)
    labels = generate_labels_from_cvrp_state(single_instance_statistics, device, epsilon)
    print("and created labels", flush=True)

    return inputs, labels


def generate_all_inputs_and_labels(alns_statistics_file, device=DEVICE):
    inputs = []
    labels = []
    """
    Retrieve the statistics saved during the ALNS execution.
    """
    alns_statistics_path = STATISTICS_DATA_PATH + alns_statistics_file
    alns_instances_statistics = retrieve_alns_stats(alns_statistics_path)
    print("{} instances in the statistics file.".format(len(alns_instances_statistics)))
    for single_instance_statistics in alns_instances_statistics:
        single_instance_inputs, single_instance_labels = \
            generate_inputs_and_labels_for_single_instance(single_instance_statistics, device)
        inputs += single_instance_inputs
        labels += single_instance_labels
        print("\t" + "-" * 15)

    return inputs, labels


def create_dataset(alns_statistics_file, device=DEVICE):
    inputs, labels = generate_all_inputs_and_labels(alns_statistics_file, device)
    dataset = CVRPDataSet(inputs, labels)
    dataset_size = len(dataset)
    train_and_val_size = int(0.8 * dataset_size)
    train_size = int(0.8 * train_and_val_size)
    torch.manual_seed(MASK_SEED)
    train__and_val_set, test_set = random_split(dataset, [train_and_val_size, dataset_size - train_and_val_size])
    train_set, val_set = random_split(train__and_val_set, [train_size, train_and_val_size - train_size])

    return train_set, val_set, test_set


def create_dataloaders(alns_statistics_file, device=DEVICE, batch_size=BATCH_SIZE, test_batch_size=BATCH_SIZE):
    train_set, val_set, test_set = create_dataset(alns_statistics_file, device)
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, collate_fn=collate)
    validation_loader = DataLoader(dataset=val_set, batch_size=test_batch_size, collate_fn=collate)
    test_loader = DataLoader(dataset=test_set, batch_size=test_batch_size, collate_fn=collate)

    return train_loader, validation_loader, test_loader
