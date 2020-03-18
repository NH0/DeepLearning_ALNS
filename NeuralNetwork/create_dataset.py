import sys
import pickle


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
        for removal_iteration in instance_statistics['Statistics']:
            x_vector.append(removal_iteration['destroyed_nodes'])
            y_vector.append(removal_iteration['objective_difference'])
    return x_vector, y_vector


if __name__ == '__main__':
    if len(sys.argv) == 2:
        x_vector, y_vector = create_dataset(sys.argv[1])
        print(y_vector.count(0.0))
    else:
        print("Usage : python create_dataset.py data_file")
