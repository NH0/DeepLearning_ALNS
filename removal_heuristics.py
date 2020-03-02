import numpy as np

degree_of_destruction = 0.2

def number_of_edges_to_remove(state):
    return int(state.instance.number_of_edges() * degree_of_destruction)

def compute_nodes_to_destroy(current_state):
    # We create a list containing the indexes of the client nodes
    # Their indexes start after ther indexes of the depots by construction
    list_of_nodes = np.arange(current_state.size) + current_state.number_of_depots

    return random_state.choice(list_of_nodes, number_of_edges_to_remove(current_state), replace=False)

def random_removal(current_state, random_state):
    destroyed = current_state.copy()

    # We choose the clients we want to remove from the instance
    nodes_to_destroy = compute_nodes_to_destroy(current_state)

    # The removal of a node N_i consists in removing the edge (N_i-1, N_i) and (N_i, N_i+1)
    # and adding the edge (N_i-1, N_i+1)
    for node in nodes_to_destroy:
        # We find the neighboring nodes
        next_node = next(destroyed.instance.neighbors(node))
        previous_node = next(destroyed.instance.predecessors(node))
        destroyed.instance.remove_edge(previous_node, node)
        destroyed.instance.remove_edge(node, next_node)
        # Avoiding useless routes
        if (next_node != previous_node):
            destroyed.instance.add_edge(previous_node, next_node)

    return destroyed
