import sys
import numpy as np

from src.ALNS.AlnsAlgorithm.solve_cvrp_alns import solve_cvrp_with_alns
from src.ALNS.AlnsStatistics.pickle_stats import pickle_alns_solution_stats
import src.ALNS.settings as settings

SIZE = settings.SIZE
CAPACITY = settings.CAPACITY
NUMBER_OF_DEPOTS = settings.NUMBER_OF_DEPOTS

ITERATIONS = settings.ITERATIONS
COLLECT_STATISTICS = settings.COLLECT_STATISTICS

NUMBER_OF_INSTANCES = settings.NUMBER_OF_INSTANCES

FILE_PATH = settings.FILE_PATH


def generate_stats(file_path=FILE_PATH, number_of_stats=NUMBER_OF_INSTANCES, add_new_stats=True):
    if number_of_stats == 1 and not add_new_stats:
        solve_cvrp_with_alns(size=SIZE, capacity=CAPACITY, number_of_depots=NUMBER_OF_DEPOTS,
                             iterations=ITERATIONS, collect_statistics=COLLECT_STATISTICS,
                             file_path=file_path, pickle_single_stat=True)
    else:
        for i in range(number_of_stats):
            seed = np.random.randint(0, 2 ** 32 - 1)
            print(" {} / {} ".format(i + 1, number_of_stats), end='')
            pickle_alns_solution_stats(result=solve_cvrp_with_alns(seed=seed, size=SIZE, capacity=CAPACITY,
                                                                   number_of_depots=NUMBER_OF_DEPOTS,
                                                                   iterations=ITERATIONS,
                                                                   collect_statistics=COLLECT_STATISTICS),
                                       file_path=file_path,
                                       file_mode='ab')


if __name__ == '__main__':
    if len(sys.argv) == 1:
        generate_stats(file_path='stats_profiling.pickle')
    elif len(sys.argv) == 2:
        generate_stats(sys.argv[1])
    elif len(sys.argv) == 3:
        generate_stats(sys.argv[1], int(sys.argv[2]))
    else:
        print("Usage : {0} [file_path] [number_of_instances]".format(sys.argv[0]))
