from alns import ALNS, State
from alns.criteria import HillClimbing

import copy

import numpy.random as rnd
import numpy.linalg as euclideanDistance

import networkx as nx
import matplotlib.pyplot as plt

from generate_instances import draw_instance, generate_cvrp_instance

SIZE = 5
CAPACITY = 40
NUMBER_OF_DEPOTS = 1

cvrp_instance = generate_cvrp_instance(SIZE, CAPACITY, NUMBER_OF_DEPOTS)
# draw_instance(cvrp_instance)
# generate_initial_solution(cvrp_instance)
# draw_instance(cvrp_instance)

class cvrpState(State):
    """
    Solution class for the CVRP problem.
    It has four data members :
     - instance : a networkx graph representing the problem. The edges on the graph are the routes used for the solution.
                    the nodes have the following attributes : coordinates, a pair of coordinates
                                                              demand, the demand of the client
                                                              isDepot, caracterizing the depots
     - size : the number of clients
     - capacity: the maximum amount of goods each vehicule can transport at a time
     - number_of_depots: the number of depot where delivery vehicules can obtain the goods to be delivered
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
            self.capacity = - demands[0]

    def copy(self):
        return copy.deepcopy(self)

    def objective(self):
        """
        WE ASSUME ALL EDGES ARE UNIQUE, NO DUPLICATES
        WE ASSUME EACH NODE HAS EXACTLY ONE OUTGOING NODE EXCEPT FOR THE DEPOTS
        """
        distance = 0
        # Start from all the depots
        for depot in range(self.number_of_depots):
            neighbors = self.instance.adj[depot]

            # Take each neighbor of the depot and follow the path
            for first_client in neighbors:

                # Distance between depot and first client
                distance += euclideanDistance.norm(self.instance.nodes[depot]['coordinates'] - self.instance.nodes[first_client]['coordinates'])

                # self.instance.adj[n] returns a dict containing all the neighbors of node n, with the keys acting as nodes
                # we keep only the first key because we assume it is the only neighbor
                outgoing_node = next(iter(self.instance.adj[first_client]))

                # distance between first client and second client
                distance += euclideanDistance.norm(self.instance.nodes[first_client]['coordinates'] - self.instance.nodes[outgoing_node]['coordinates'])

                # If the second client isn't a depot, we follow the path
                # Else, we get to the next path
                while not self.instance.nodes[outgoing_node]['isDepot']:
                    second_outgoing_node = next(iter(self.instance.adj[outgoing_node]))
                    distance += euclideanDistance.norm(self.instance.nodes[outgoing_node]['coordinates'] - self.instance.nodes[second_outgoing_node]['coordinates'])
                    outgoing_node = second_outgoing_node

        return distance

    def draw(self, show_demands=False):
        """
        Draws the instance.
        Parameters :
        instance : the CVRP to draw
        show_demands : show each client's demand as node labels
        """

        depots = [i for i in range(self.number_of_depots)]
        position = nx.get_node_attributes(self.instance, 'coordinates')

        if (show_demands):
            # To label with demands, use labels=demands and with_label=True
            demands = nx.get_node_attributes(self.instance, 'demand')
            nx.draw(self.instance, position, nodelist=[i + self.number_of_depots for i in range(self.size)], labels=demands, node_size=50, node_color='blue', node_shape='o')
        else:
            nx.draw(self.instance, position, nodelist=[i + self.number_of_depots for i in range(self.size)], node_size=50, node_color='blue', node_shape='o')

        nx.draw(self.instance, position, nodelist=depots, node_size=50, node_color='red', node_shape='d')
        plt.show()

def generate_initial_solution(cvrp_state):
    """
    Generates a solution where the delivery vehicules returns to the depot after each client.
    Parameters :
    cvrp_state : the instance of the CVRP state that is to be solved
    """
    cvrp_state.instance = nx.create_empty_copy(cvrp_state.instance)

    edges = [(0, i + cvrp_state.number_of_depots) for i in range(cvrp_state.size)] + [(i + cvrp_state.number_of_depots, 0) for i in range(cvrp_state.size)]

    cvrp_state.instance.add_edges_from(edges)

    return cvrp_state

state = cvrpState(cvrp_instance, size=SIZE, number_of_depots=NUMBER_OF_DEPOTS, capacity=CAPACITY)
state.draw()
initial_solution = generate_initial_solution(state)
initial_solution.draw()
initial_distance = initial_solution.objective()
print("Initial distance is ",initial_distance)
