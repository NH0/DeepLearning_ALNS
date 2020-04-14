import pickle

from sys import argv
from copy import deepcopy
from numpy import reshape
from pprint import pprint

from ALNS.alns_state import CvrpState
from ALNS.generate_instances import generate_cvrp_instance


def retrieve_alns_stats(file):
    statistics = []
    with open(file, 'rb') as file_alns_data:
        while True:
            try:
                statistics.append(pickle.load(file_alns_data))
            # pickling is done one object at a time
            except EOFError:
                break
            except pickle.UnpicklingError:
                print('\n---------\nError : Could not retrieve the data in {0}.\n---------'.format(file))
                exit(1)
        return statistics


def create_dataset(file):
    statistics = retrieve_alns_stats(file)
    x_vector = []
    y_vector = []
    for instance_statistics in statistics:
        size = instance_statistics['Size']
        number_of_depots = instance_statistics['Number_of_depots']
        number_of_nodes = size + number_of_depots
        seed = instance_statistics['Seed']

        # Capacity isn't necessary
        alns_instance = generate_cvrp_instance(size, number_of_depots=number_of_depots, seed=seed)
        alns_state = CvrpState(alns_instance, collect_alns_statistics=False,
                               size=size, number_of_depots=number_of_depots)
        distances = alns_state.distances
        instance_state_structure = [[[0, distances[i][j]]
                                     for j in range(number_of_nodes)]
                                    for i in range(number_of_nodes)]

        for removal_iteration in instance_statistics['Statistics']:
            removal_state = deepcopy(instance_state_structure)
            for previous_node, successor_node, _ in removal_iteration['list_of_edges']:
                removal_state[previous_node][successor_node][0] = 1

            consecutive_inputs = []
            for node in removal_iteration['destroyed_nodes']:
                one_input = [[[0, 0] if i != node else [1, 0] for i in range(number_of_nodes)]]
                one_input = one_input + removal_state
                one_input = reshape(one_input, 2 * (number_of_nodes * (number_of_nodes + 1)))
                consecutive_inputs.append(one_input)

            x_vector.append(consecutive_inputs)
            y_vector.append(removal_iteration['objective_difference'])
    return x_vector, y_vector


if __name__ == '__main__':
    if len(argv) == 2:
        x_vector, y_vector = create_dataset(argv[1])
    else:
        print("Usage : python create_dataset.py data_file")
