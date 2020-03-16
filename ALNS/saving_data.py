import pickle

FILE_PATH = "../data/alns_stats.pickle"


def save_alns_solution_stats(result, file_path=FILE_PATH) -> None:
    with open(file_path, 'ab') as file_alns_stats:
        try:
            pickle.dump(result, file_alns_stats)
            print('Successfully saved the data in {0}'.format(file_path))
        except (pickle.PicklingError, AttributeError):
            print('\n---------\nError : Wrong data format. Nothing was saved.\n---------')
            exit(1)
