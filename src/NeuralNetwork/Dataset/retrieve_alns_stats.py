import pickle
import copy

import src.NeuralNetwork.parameters as parameters
from src.ALNS.AlnsStatistics.pickle_stats import pickle_alns_solution_stats

DATA_PATH = parameters.STATISTICS_DATA_PATH
EPSILON = parameters.EPSILON


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


def create_evenly_distributed_alns_stats(input_pickle_path, output_pickle_path, epsilon=EPSILON):
    statistics = retrieve_alns_stats(input_pickle_path)
    if len(statistics) != 1:
        print("Error : statistics contain more than one alns execution ! Using only the first execution.")
    statistics = statistics[0]
    number_of_nonnull_values = 0
    for iteration in statistics['Statistics']:
        if abs(iteration['objective_difference']) > epsilon:
            number_of_nonnull_values += 1

    dataset = copy.deepcopy(statistics)
    dataset['Statistics'].clear()
    for iteration in statistics['Statistics']:
        if abs(iteration['objective_difference']) > epsilon:
            dataset['Statistics'].append(iteration)
        elif abs(iteration['objective_difference']) <= epsilon and number_of_nonnull_values > 0:
            dataset['Statistics'].append(iteration)
            number_of_nonnull_values -= 1

    pickle_alns_solution_stats(dataset, output_pickle_path)


if __name__ == '__main__':
    create_evenly_distributed_alns_stats(
        DATA_PATH + 'stats_1000iter.pickle',
        DATA_PATH + '50-50_stats_1000iter.pickle')
