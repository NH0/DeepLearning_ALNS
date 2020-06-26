import torch

import src.NeuralNetwork.parameters as parameters

from dgl import batch as dgl_batch
from torch.utils.data.dataset import random_split
from src.ALNS.CVRP.CVRP import CvrpState
from src.ALNS.CVRP.generate_cvrp_graph import generate_cvrp_instance
from src.ALNS.CVRP.from_nx_to_dgl import make_complete_nx_graph, generate_dgl_graph, initialize_dgl_features
from src.NeuralNetwork.Dataset.retrieve_alns_stats import retrieve_alns_stats
from src.NeuralNetwork.Dataset.dataset import CVRPDataSet

STATISTICS_DATA_PATH = parameters.STATISTICS_DATA_PATH
ALNS_STATISTICS_FILE = parameters.ALNS_STATISTICS_FILE
INPUTS_LABELS_PATH = parameters.INPUTS_LABELS_PATH
MODEL_PARAMETERS_PATH = parameters.MODEL_PARAMETERS_PATH

EPSILON = parameters.EPSILON

MASK_SEED = parameters.MASK_SEED
BATCH_SIZE = parameters.BATCH_SIZE

DEVICE = parameters.DEVICE


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
    make_complete_nx_graph(initial_state.instance)

    return initial_state


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
        [generate_dgl_graph(nx_graph) for _ in alns_instance_statistics['Statistics']]
    # All graphs must have the same number of node
    # They actually represent the same CVRP problem with different solution
    # The data is the CVRP state at different iterations of the ALNS heuristic

    for i, graph in enumerate(list_of_dgl_graphs):
        initialize_dgl_features(cvrp_state,
                                graph,
                                alns_instance_statistics['Statistics'][i]['destroyed_nodes'],
                                alns_instance_statistics['Statistics'][i]['list_of_edges'],
                                device)

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


def create_dataset(inputs, labels, validation_set=False):
    dataset = CVRPDataSet(inputs, labels)
    dataset_size = len(dataset)
    train_and_val_size = int(0.8 * dataset_size)
    train_size = int(0.8 * train_and_val_size)
    torch.manual_seed(MASK_SEED)
    train_and_val_set, test_set = random_split(dataset, [train_and_val_size, dataset_size - train_and_val_size])
    if validation_set:
        train_set, val_set = random_split(train_and_val_set, [train_size, train_and_val_size - train_size])

        return train_set, val_set, test_set

    return train_and_val_set, test_set


def collate(sample):
    graphs, labels = map(list, zip(*sample))
    graph_batch = dgl_batch(graphs)
    return graph_batch, torch.tensor(labels, device=labels[0].device)
