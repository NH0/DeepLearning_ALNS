import pickle

import src.ALNS.settings as settings

FILE_PATH = settings.FILE_PATH
FILE_MODE = settings.FILE_MODE


def pickle_alns_solution_stats(result, file_path=FILE_PATH, file_mode=FILE_MODE) -> None:
    with open(file_path, file_mode) as file_alns_stats:
        try:
            pickle.dump(result, file_alns_stats)
            print('Successfully saved the data in {0} using mode {1}'.format(file_path, file_mode))
        except (pickle.PicklingError, AttributeError):
            print('\n---------\nError : Wrong data format. Nothing was saved.\n---------')
            exit(1)
