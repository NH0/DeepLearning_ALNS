from alns import ALNS
from alns.criteria import SimulatedAnnealing

import numpy.random as rnd
from numpy import log

from src.ALNS.CVRP.generate_cvrp_graph import generate_cvrp_instance
from src.ALNS.AlnsAlgorithm.removal_heuristics import removal_heuristic
from src.ALNS.AlnsAlgorithm.repair_heuristics import greedy_insertion
from src.ALNS.AlnsStatistics.pickle_stats import pickle_alns_solution_stats
from src.ALNS.CVRP.CVRP import CvrpState, generate_initial_solution

import src.ALNS.settings as settings

SEED = settings.SEED

SIZE = settings.SIZE
CAPACITY = settings.CAPACITY
NUMBER_OF_DEPOTS = settings.NUMBER_OF_DEPOTS

DEGREE_OF_DESTRUCTION = settings.DEGREE_OF_DESTRUCTION
DETERMINISM = settings.DETERMINISM

ITERATIONS = settings.NUMBER_OF_DEPOTS
WEIGHTS = settings.WEIGHTS
OPERATOR_DECAY = settings.OPERATOR_DECAY
COLLECT_STATISTICS = settings.COLLECT_STATISTICS


def compute_initial_temperature(initial_solution_cost, start_temperature_control):
    return - initial_solution_cost * start_temperature_control / log(0.5)


def solve_cvrp_with_alns(seed=SEED, size=SIZE, capacity=CAPACITY, number_of_depots=NUMBER_OF_DEPOTS,
                         iterations=ITERATIONS, collect_statistics=COLLECT_STATISTICS, **kwargs):
    weights = WEIGHTS
    operator_decay = OPERATOR_DECAY
    start_temperature_control = settings.START_TEMPERATURE_CONTROL
    cooling_rate = settings.COOLING_RATE
    end_temperature = settings.END_TEMPERATURE

    if 'weights' in kwargs:
        weights = kwargs['weights']
    if 'operator_decay' in kwargs:
        operator_decay = kwargs['operator_decay']
    if 'start_temperature_control' in kwargs:
        start_temperature_control = kwargs['start_temperature_control']
    if 'cooling_rate' in kwargs:
        cooling_rate = kwargs['cooling_rate']
    if 'end_temperature' in kwargs:
        end_temperature = kwargs['end_temperature']

    cvrp_instance = generate_cvrp_instance(size, capacity, number_of_depots, seed)
    # Create an empty state
    initial_state = CvrpState(cvrp_instance, collect_alns_statistics=collect_statistics, size=size,
                              number_of_depots=number_of_depots,
                              capacity=capacity)
    initial_solution = generate_initial_solution(initial_state)
    initial_distance = initial_solution.objective()

    # Initialize ALNS
    random_state = rnd.RandomState(seed)
    alns = ALNS(random_state)
    alns.add_destroy_operator(removal_heuristic)
    alns.add_repair_operator(greedy_insertion)

    initial_temperature = compute_initial_temperature(initial_distance, start_temperature_control)
    criterion = SimulatedAnnealing(initial_temperature,
                                   end_temperature,
                                   cooling_rate)

    # Solve the cvrp using ALNS
    result = alns.iterate(initial_solution, weights, operator_decay, criterion, iterations=iterations,
                          collect_stats=collect_statistics)
    solution = result.best_state

    # Create the statistics if necessary
    solution_data = {}
    if solution.collect_alns_statistics:
        solution_data = {'Size': solution.size,
                         'Number_of_depots': solution.number_of_depots,
                         'Capacity': solution.capacity,
                         'Seed': seed,
                         'Parameters': {'decay': operator_decay,
                                        'degree_destruction': DEGREE_OF_DESTRUCTION,
                                        'determinism': DETERMINISM}}
        solution_statistics = [{'destroyed_nodes': solution.statistics['destroyed_nodes'][i],
                                'objective_difference': result.statistics.objectives[i + 1]
                                                        - result.statistics.objectives[i],
                                'list_of_edges': solution.statistics['list_of_edges'][i]}
                               for i in range(iterations)]
        solution_data['Statistics'] = solution_statistics

        if 'pickle_single_stat' in kwargs and kwargs['pickle_single_stat']:
            if 'file_path' in kwargs:
                pickle_alns_solution_stats(solution_data, file_path=kwargs['file_path'])
            else:
                pickle_alns_solution_stats(solution_data)

    return solution_data


if __name__ == '__main__':
    initial_instance = generate_cvrp_instance(SIZE, CAPACITY, NUMBER_OF_DEPOTS, SEED)
    void_state = CvrpState(initial_instance, collect_alns_statistics=False, size=SIZE,
                           number_of_depots=NUMBER_OF_DEPOTS,
                           capacity=CAPACITY)
    main_initial_solution = generate_initial_solution(void_state)
    main_initial_solution.draw()
