import pickle


def retrieve_alns_stats(file):
    with open(file, 'rb') as file_alns_data:
        try:
            objectives = pickle.load(file_alns_data)
            print(objectives)
        except pickle.UnpicklingError:
            print('\n---------\nError : Could not retrieve the data.\n---------')
            exit(1)


retrieve_alns_stats("../data/alns_stats.pickle")
