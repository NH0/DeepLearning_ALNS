import os
import sys
import numpy as np

sys.path.append(os.getcwd())

from ALNS.execute_alns import solve_cvrp_with_alns
import ALNS.settings as settings

SIZE = settings.SIZE
CAPACITY = settings.CAPACITY
NUMBER_OF_DEPOTS = settings.NUMBER_OF_DEPOTS

ITERATIONS = settings.ITERATIONS
COLLECT_STATISTICS = settings.COLLECT_STATISTICS

NUMBER_OF_INSTANCES = settings.NUMBER_OF_INSTANCES


def generate_stats(file_path, number_of_stats=NUMBER_OF_INSTANCES):
    for i in range(number_of_stats):
        seed = np.random.randint(0, 2 ** 32 - 1)
        solve_cvrp_with_alns(seed=seed, size=SIZE, capacity=CAPACITY, number_of_depots=NUMBER_OF_DEPOTS,
                             iterations=ITERATIONS, collect_statistics=COLLECT_STATISTICS, file_path=file_path)


if __name__ == '__main__':
    if len(sys.argv) == 2:
        generate_stats(sys.argv[1])
    elif len(sys.argv) == 3:
        generate_stats(sys.argv[1], int(sys.argv[2]))
    else:
        print("Usage : generate_stats.py file_path [number_of_instances]")
