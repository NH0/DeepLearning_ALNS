import pickle
import os
import copy

from ALNS.AlnsStatistics.pickle_stats import pickle_alns_solution_stats

DATA_PATH = os.getcwd().rpartition('/')[0] + '/data/'
EPSILON = 0.00001


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
        DATA_PATH + '1inst_50nod_40cap_1dep_50000iter_0.8decay_0.35destr_18determ.pickle',
        DATA_PATH + 'dataset_50-50_1inst_50nod_40cap_1dep_50000iter_0.8decay_0.35destr_18determ.pickle')
