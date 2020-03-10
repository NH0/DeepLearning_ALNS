import numpy.linalg as euclideanDistance
import numpy as np

def compute_adjacency_matrix(state):
    number_of_nodes = state.size + state.number_of_depots

    adjacency_matrix = np.zeros((number_of_nodes, number_of_nodes))

    for i in range(number_of_nodes):
        for j in range(i, number_of_nodes):
            adjacency_matrix[i][j] = euclideanDistance.norm(state.instance.nodes[i]['coordinates'] - state.instance.nodes[j]['coordinates'])

    # Normalization in [0,1]
    max_distance = np.amax(adjacency_matrix)
    min_distance = np.amin(adjacency_matrix)

    for i in range(number_of_nodes):
        for j in range(i, number_of_nodes):
            adjacency_matrix[i][j] = (adjacency_matrix[i][j] - min_distance) / (max_distance - min_distance)
            adjacency_matrix[j][i] = adjacency_matrix[i][j]

    return adjacency_matrix

def compute_single_route_distance(state, start_depot, first_client):
    distance = 0
    # Distance between depot and first client
    distance += state.distances[start_depot][first_client]

    # state.instance.neighbors[n] returns an iterators over the successors of the node
    # we keep only the first neighbor because we assume their is only one
    next_node = next(state.instance.neighbors(first_client))

    # distance between first client and second client
    distance += state.distances[first_client][next_node]

    # If the second client isn't a depot, we follow the path
    # Else, we get to the next path
    while not state.instance.nodes[next_node]['isDepot']:
        second_next_node = next(state.instance.neighbors(next_node))
        distance += state.distances[next_node][second_next_node]
        next_node = second_next_node

    return distance

def compute_route_demand(state, start_depot, first_client):
    demand = state.instance.nodes[first_client]['demand']

    next_node = next(state.instance.neighbors(first_client))

    while not state.instance.nodes[next_node]['isDepot']:
        demand = state.instance.nodes[next_node]['demand']
        next_node = next(state.instance.neighbors(next_node))

    return demand

def compute_defined_insertion_cost(state, previous_node, next_node, node_to_insert):
    return ( state.distances[previous_node][node_to_insert]
           + state.distances[node_to_insert][next_node]
           - state.distances[previous_node][next_node] )

def compute_route_best_insertion_cost(state, start_depot, first_client, node_to_insert):
    if (compute_route_demand(state, start_depot, first_client) + state.instance.nodes[node_to_insert]['demand'] > state.capacity):
        return (float("inf"), (-1,-1))

    best_insertion_cost = compute_defined_insertion_cost(state, start_depot, first_client, node_to_insert)
    best_insertion_nodes = (start_depot, first_client)

    # To avoid confusion during the loop
    previous_node = first_client
    next_node = next(state.instance.neighbors(previous_node))

    while True:
        insertion_cost = compute_defined_insertion_cost(state, previous_node, next_node, node_to_insert)
        if (insertion_cost < best_insertion_cost):
            best_insertion_cost = insertion_cost
            best_insertion_nodes = (previous_node, next_node)

        if not state.instance.nodes[next_node]['isDepot']:
            break
        previous_node = next_node
        next_node = next(state.instance.neighbors(previous_node))

    return (best_insertion_cost, best_insertion_nodes)

def compute_minimum_cost_position(state, node_to_insert):
    minimum_cost = float("inf")
    minimum_cost_nodes = (0, 1)
    for depot in range(state.number_of_depots):
        for first_client in state.instance.neighbors(depot):
            cost, nodes = compute_route_best_insertion_cost(state, depot, first_client, node_to_insert)
            if (cost < minimum_cost):
                minimum_cost = cost
                minimum_cost_nodes = nodes
    return (minimum_cost, minimum_cost_nodes)
