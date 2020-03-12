import numpy.linalg as euclidean_distance
import numpy as np

from ALNS.execute_alns import CvrpState


def compute_adjacency_matrix(state: CvrpState) -> np.ndarray:
    """
    Computes the adjacency matrix of a given graph using the euclidean distance.
    Parameters
    ----------
    state : a CvrpState instance

    Returns
    -------
    The adjacency matrix of the nodes
    """
    number_of_nodes = state.size + state.number_of_depots

    adjacency_matrix = np.zeros((number_of_nodes, number_of_nodes))

    for i in range(number_of_nodes):
        for j in range(i, number_of_nodes):
            adjacency_matrix[i][j] = euclidean_distance.norm(
                state.instance.nodes[i]['coordinates'] - state.instance.nodes[j]['coordinates'])

    # Normalization of the distances in [0,1]
    max_distance = np.amax(adjacency_matrix)
    min_distance = np.amin(adjacency_matrix)

    for i in range(number_of_nodes):
        for j in range(i, number_of_nodes):
            adjacency_matrix[i][j] = (adjacency_matrix[i][j] - min_distance) / (max_distance - min_distance)
            adjacency_matrix[j][i] = adjacency_matrix[i][j]

    return adjacency_matrix


def compute_closest_depot(state: CvrpState, node: int) -> int:
    """
    Given a node, finds the closest depot to the node.
    Parameters
    ----------
    state : a CvrpState instance.
    node : the index of the node.

    Returns
    -------
    The index of the closest depot.
    """
    closest = 0
    for depot in range(state.number_of_depots):
        if state.distances[depot][node] < state.distances[closest][node]:
            closest = depot
    return closest


def compute_single_route_distance(state: CvrpState, start_depot: int, first_client: int) -> float:
    """
    Given a route characterized by the depot it is linked to and the first client it visits, compute the distance of
    the route.
    Parameters
    ----------
    state : a CvrpState instance.
    start_depot : the index of the depot linked to the route.
    first_client : the index of the first node visited after the depot (successor of the depot).

    Returns
    -------
    An floating value representing the distance of the route.
    """
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


def compute_route_demand(state: CvrpState, first_client: int) -> float:
    """
    Given a route characterized by the first client it visits, we compute the total demand of the route.
    That is the sum of the demands of the clients visited during the route.
    Parameters
    ----------
    state : a CvrpState instance.
    first_client : the index of the first node visited after the depot (successor of the depot).

    Returns
    -------
    A floating value representing the total demand of the route
    """
    demand = state.instance.nodes[first_client]['demand']

    next_node = next(state.instance.neighbors(first_client))

    while not state.instance.nodes[next_node]['isDepot']:
        demand += state.instance.nodes[next_node]['demand']
        next_node = next(state.instance.neighbors(next_node))

    return demand


def compute_defined_insertion_cost(state: CvrpState, previous_node: int, next_node: int, node_to_insert: int) -> float:
    """
    Compute the cost of inserted a node between two connected nodes.
    Parameters
    ----------
    state : a CvrpState instance.
    previous_node : the index of one of the connected nodes. The predecessor of the other connected node.
    next_node : the index of one of the connected nodes. The successor of the other connected node.
    node_to_insert : the index node to insert between the previous_node and the next_node.

    Returns
    -------
    float containing the cost of inserting the node.
    """
    return (state.distances[previous_node][node_to_insert]
            + state.distances[node_to_insert][next_node]
            - state.distances[previous_node][next_node])


def compute_route_best_insertion_cost(state: CvrpState, start_depot: int, first_client: int, node_to_insert: int) \
        -> (float, (int, int)):
    """
    Compute the best place to insert a given node in a given route.
    The route is uniquely defined by the depot it is linked to and the first visited client.
    Parameters
    ----------
    state : a CvrpState instance.
    start_depot : the index of the depot linked to the route.
    first_client : the index of the first node visited after the depot (successor of the depot).
    node_to_insert : the index of the node to insert between the previous_node and the next_node.

    Returns
    -------
    (float, (int, int)) representing the best insertion possible in the route with
        (cost of insertion, (previous node, successor node))
    """
    # If a route cannot accept the demand then infinite cost
    if (compute_route_demand(state, first_client)
            + state.instance.nodes[node_to_insert]['demand'] > state.capacity):
        return float("inf"), (-1, -1)

    best_insertion_cost = compute_defined_insertion_cost(state, start_depot, first_client, node_to_insert)
    best_insertion_nodes = (start_depot, first_client)

    # To avoid confusion during the loop
    previous_node = first_client
    next_node = next(state.instance.neighbors(previous_node))

    # Loop until the end of the route. Using infinite loop because the structure do {...} while (...) doesn't exist
    while True:
        insertion_cost = compute_defined_insertion_cost(state, previous_node, next_node, node_to_insert)
        if insertion_cost < best_insertion_cost:
            best_insertion_cost = insertion_cost
            best_insertion_nodes = (previous_node, next_node)

        if state.instance.nodes[next_node]['isDepot']:
            break
        previous_node = next_node
        next_node = next(state.instance.neighbors(previous_node))

    return best_insertion_cost, best_insertion_nodes


def compute_minimum_cost_position(state: CvrpState, node_to_insert: int) -> (float, (int, int)):
    """
    Compute the best edge to insert a node into.
    Parameters
    ----------
    state : a CvrpState instance.
    node_to_insert : the index of the node to insert in the solution.

    Returns
    -------
    (float, (int, int)) representing the best insertion possible in the solution with
        (cost of insertion, (previous node, successor node))
    """
    minimum_cost = float("inf")
    minimum_cost_nodes = (0, 1)
    for depot in range(state.number_of_depots):
        # Loop trough all routes from the given depot
        for first_client in state.instance.neighbors(depot):
            cost, nodes = compute_route_best_insertion_cost(state, depot, first_client, node_to_insert)
            if cost < minimum_cost:
                minimum_cost = cost
                minimum_cost_nodes = nodes
    # If all routes are full then link the node to the first depot (back and forth route)
    if minimum_cost == float("inf"):
        closest_depot = compute_closest_depot(state, node_to_insert)

        return 2 * state.distances[closest_depot][node_to_insert], (0, 0)

    return minimum_cost, minimum_cost_nodes


def print_routes_demands(state: CvrpState) -> None:
    """
    Print the total demands for each route in the instance.
    Parameters
    ----------
    state : a CvrpState instance.
    """
    for depot in range(state.number_of_depots):
        for first_node in state.instance.neighbors(depot):
            print("Route starting with {0} has a cost of {1}"
                  .format(first_node, compute_route_demand(state, first_node)))
