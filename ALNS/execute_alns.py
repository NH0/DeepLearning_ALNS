from alns import ALNS, State
from alns.criteria import HillClimbing

import copy

import numpy.random as rnd

import networkx as nx
import matplotlib.pyplot as plt

from ALNS.generate_instances import generate_cvrp_instance
from ALNS.compute_distances import compute_single_route_distance, compute_adjacency_matrix, print_routes_demands
from ALNS.removal_heuristics import removal_heuristic
from ALNS.repair_heuristics import greedy_insertion
from ALNS.saving_data import save_alns_solution_stats

SEED = 2020

SIZE = 40
CAPACITY = 40
NUMBER_OF_DEPOTS = 1

cvrp_instance = generate_cvrp_instance(SIZE, CAPACITY, NUMBER_OF_DEPOTS, SEED)


# for i in range(SIZE):
#     print(cvrp_instance.nodes[i])
# draw_instance(cvrp_instance)
# generate_initial_solution(cvrp_instance)
# draw_instance(cvrp_instance)

class CvrpState(State):
    """
    Solution class for the CVRP problem.
    It has four data members :
     - instance : a networkx graph representing the problem. The edges on the graph are the routes used for the solution.
                    the nodes have the following attributes : coordinates, a pair of coordinates
                                                              demand, the demand of the client
                                                              isDepot, characterizing the depots
     - size : the number of clients
     - capacity: the maximum amount of goods each vehicule can transport at a time
     - number_of_depots: the number of depot where delivery vehicules can obtain the goods to be delivered
     - distances: the adjacency matrix between all nodes
    """

    def __init__(self, instance, **parameters):
        self.instance = instance

        if 'number_of_depots' in parameters:
            self.number_of_depots = parameters['number_of_depots']
        else:
            number_of_nodes = instance.number_of_nodes()
            are_depots = nx.get_node_attributes(instance, 'isDepot')
            depots = []
            for i in range(number_of_nodes):
                if are_depots[i]:
                    depots.append(i)
                else:
                    break
            number_of_depots = len(depots)
            self.number_of_depots = number_of_depots

        if 'size' in parameters:
            self.size = parameters['size']
        else:
            number_of_nodes = instance.number_of_nodes()
            self.size = number_of_nodes - self.number_of_depots

        if 'capacity' in parameters:
            self.capacity = parameters['capacity']
        else:
            demands = nx.get_node_attributes(instance, 'demand')
            # The first demand is the demand of the first depot
            # Its value is - capacity by definition
            self.capacity = - demands[0]

        self.distances = compute_adjacency_matrix(self)

    def copy(self):
        return copy.deepcopy(self)

    def objective(self):
        """
        WE ASSUME ALL EDGES ARE UNIQUE, NO DUPLICATES
        WE ASSUME EACH NODE HAS EXACTLY ONE SUCCESSOR (NEXT) NODE EXCEPT FOR THE DEPOTS
        """
        distance = 0
        # Start from all the depots
        for depot in range(self.number_of_depots):
            neighbors = self.instance.neighbors(depot)

            # Take each neighbor of the depot and follow the path
            for first_client in neighbors:
                distance += compute_single_route_distance(self, depot, first_client)

        return distance

    def draw(self, show_demands=False):
        """
        Draws the instance.
        Parameters :
        - instance : the CVRP to draw
        - show_demands : show each client's demand as node labels
        """

        depots = [i for i in range(self.number_of_depots)]
        position = nx.get_node_attributes(self.instance, 'coordinates')

        if show_demands:
            # To label with demands, use labels=demands and with_label=True
            demands = nx.get_node_attributes(self.instance, 'demand')
            nx.draw(self.instance, position, nodelist=[i + self.number_of_depots for i in range(self.size)],
                    labels=demands, node_size=20, node_color='blue', node_shape='o')
        else:
            # We display the node indexes on the coordinates of the nodes
            nx.draw(self.instance, position, nodelist=[i + self.number_of_depots for i in range(self.size)],
                    with_labels=True, node_size=20, node_color='#B2E3C4', node_shape='o', font_weight='bold')

        nx.draw(self.instance, position, nodelist=depots, node_size=50, node_color='red', node_shape='d')
        plt.show()


def generate_initial_solution(cvrp_state):
    """
    Generates a greedy solution.
    Starts from the first depot, and finds the closest nodes and links it to the depot.
    Then from this node and so on finds the closest unvisited node until the route has the maximum capacity.
    Starts again from the first depot and create the same way a second route,
        and so on until all nodes have been visited.
    Parameters :
    cvrp_state : the instance of the CVRP state that is to be solved
    """
    cvrp_state.instance = nx.create_empty_copy(cvrp_state.instance)
    edges = []

    # List of the indexes of unvisited nodes, updated each time a node is added to a route
    unvisited_nodes = [i + cvrp_state.number_of_depots for i in range(cvrp_state.size)]
    first_node = 0

    # Add every node in a route
    while len(unvisited_nodes) > 0:
        route_demand = 0
        # This loops for each new route
        while len(unvisited_nodes) > 0:
            closest_node = sorted(unvisited_nodes, key=lambda node: cvrp_state.distances[first_node][node])[0]
            # Ensure the node can be added to the route
            if route_demand + cvrp_state.instance.nodes[closest_node]['demand'] < cvrp_state.capacity:
                edges.append((first_node, closest_node))
                unvisited_nodes.remove(closest_node)
                route_demand += cvrp_state.instance.nodes[closest_node]['demand']
            # Start a new route
            else:
                break
            first_node = closest_node
        # Close the previous route be connecting the last node and the first depot
        edges.append((first_node, 0))
        # Start again from the first depot
        first_node = 0

    cvrp_state.instance.add_edges_from(edges)

    return cvrp_state

# Create an empty state
void_state = CvrpState(cvrp_instance, size=SIZE, number_of_depots=NUMBER_OF_DEPOTS, capacity=CAPACITY)
# void_state.draw()
initial_solution = generate_initial_solution(void_state)
initial_solution.draw()
initial_distance = initial_solution.objective()
print("Initial distance is ", initial_distance)

# Initialize ALNS
random_state = rnd.RandomState(SEED)
alns = ALNS(random_state)
alns.add_destroy_operator(removal_heuristic)
alns.add_repair_operator(greedy_insertion)
criterion = HillClimbing()

# Solve the cvrp using ALNS
result = alns.iterate(initial_solution, [3, 2, 1, 0.5], 0.8, criterion, iterations=100, collect_stats=True)

solution = result.best_state
print("Optimal distance is ", solution.objective())
print_routes_demands(solution)
solution.draw()

# result.plot_objectives()
plt.show()

save_alns_solution_stats(result.statistics.objectives)
