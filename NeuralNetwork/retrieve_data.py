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


if __name__ == '__main__':
    if len(sys.argv) == 2:
        retrieve_alns_stats(sys.argv[1])
    else:
        print("Usage : python retrieve_data.py data_file")
