import pickle

FILE_PATH = "../data/alns_stats.pickle"


def save_alns_solution_stats(result) -> None:
    with open(FILE_PATH, 'ab') as file_alns_stats:
        try:
            pickle.dump(result, file_alns_stats)
        except (pickle.PicklingError, AttributeError):
            print('\n---------\nError : Wrong data format. Nothing was saved.\n---------')
            exit(1)
