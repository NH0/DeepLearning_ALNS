import os
import copy

from NeuralNetwork.create_dataset import retrieve_alns_stats
from ALNS.saving_data import save_alns_solution_stats

DATA_PATH = os.getcwd().rpartition('/')[0] + '/data/'
EPSILON = 0.00001


def create_data_from_pickle(input_pickle_path, output_pickle_path, epsilon=EPSILON):
    statistics = retrieve_alns_stats(input_pickle_path)
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

    save_alns_solution_stats(dataset, output_pickle_path)


if __name__ == '__main__':
    create_data_from_pickle(DATA_PATH + '1inst_50nod_40cap_1dep_50000iter_0.8decay_0.35destr_18determ.pickle',
                            DATA_PATH + 'dataset_50-50_'
                                        '1inst_50nod_40cap_1dep_50000iter_0.8decay_0.35destr_18determ.pickle')
