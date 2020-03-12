import networkx as nx
import numpy as np

import matplotlib.pyplot as plt


def generate_cvrp_instance(size=50, capacity=40, number_of_depots=1, SEED=2020):
    """
    Generates an instance of the Capacitated Vehicle Routing Problem.
    Parameters :
    size: the number of clients
    capacity: the maximum amount of goods each vehicle can transport at a time
    number_of_depots: the number of depot where delivery vehicles can obtain the goods to be delivered
    """
    np.random.seed(SEED)

    # The first elements are the depots, the following the clients
    random_nodes_coordinates = np.random.uniform(0, 1, (number_of_depots + size, 2))

    # We only generate random demand for the clients
    random_nodes_demands = np.random.uniform(1, capacity / 5, size)

    # Nodes have three attributes : coordinates, demand (which is negative for the depot) and a boolean to know if it
    # is a depot
    nodes = [(i,
              {'coordinates': random_nodes_coordinates[i],
               'demand': - capacity if i < number_of_depots else random_nodes_demands[i - number_of_depots],
               'isDepot': i < number_of_depots})
             for i in range(number_of_depots + size)]

    instance = nx.DiGraph()
    instance.add_nodes_from(nodes)

    return instance


def draw_instance(instance, show_demands=False):
    """
    Draws an instance.
    Parameters :
    instance : the CVRP to draw
    show_demands : show each client's demand as node labels
    """

    number_of_nodes = instance.number_of_nodes()
    position = nx.get_node_attributes(instance, 'coordinates')
    # To label with demands, use labels=demands and with_label=True
    # demands = nx.get_node_attributes(instance, 'demand')
    are_depots = nx.get_node_attributes(instance, 'isDepot')
    depots = [i for i in range(number_of_nodes) if are_depots[i]]

    nx.draw(instance, position, nodelist=depots, node_size=50, node_color='red', node_shape='d')
    nx.draw(instance, position, nodelist=[i for i in range(len(depots), number_of_nodes)], node_size=50,
            node_color='blue', node_shape='o')
    plt.show()


if __name__ == '__main__':
    for i in range(3):
        draw_instance(generate_cvrp_instance(50, 40, 3))
