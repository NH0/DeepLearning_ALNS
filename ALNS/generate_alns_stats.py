import numpy as np
from ALNS.execute_alns import solve_cvrp_with_alns

SIZE = 30
CAPACITY = 40
NUMBER_OF_DEPOTS = 1

ITERATIONS = 10000
COLLECT_STATISTICS = True

NUMBER_OF_INSTANCES = 10000


def generate_stats(file_path, number_of_stats=10000):
    for i in range(number_of_stats):
        seed = np.random.randint(0, 2**32 - 1)
        solve_cvrp_with_alns(seed=seed, size=SIZE, capacity=CAPACITY, number_of_depots=NUMBER_OF_DEPOTS,
                             iterations=ITERATIONS, collect_statistics=COLLECT_STATISTICS, file_path=file_path)


generate_stats('/tmp/data.pickle', NUMBER_OF_INSTANCES)
