import os
import sys

sys.path.append(os.getcwd())

import numpy as np
from ALNS.execute_alns import solve_cvrp_with_alns

SIZE = 50
CAPACITY = 40
NUMBER_OF_DEPOTS = 1

ITERATIONS = 30000
COLLECT_STATISTICS = True

NUMBER_OF_INSTANCES = 1


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
