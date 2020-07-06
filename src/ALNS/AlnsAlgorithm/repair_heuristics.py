from src.ALNS.AlnsAlgorithm.compute_distances import compute_minimum_cost_position
from src.ALNS.CVRP.from_nx_to_dgl import edge_ends_to_edge_index


def greedy_insertion(current_state, random_state):

    unvisited_nodes = [node for node in current_state.instance.nodes() if current_state.instance.degree(node) == 0]
    number_of_nodes = current_state.instance.number_of_nodes()

    while len(unvisited_nodes) > 0:
        best_node = unvisited_nodes[0]
        best_node_cost, best_position = compute_minimum_cost_position(current_state, best_node)
        for node in unvisited_nodes[1:]:
            node_cost, position = compute_minimum_cost_position(current_state, node)
            if node_cost < best_node_cost:
                best_node = node
                best_node_cost = node_cost
                best_position = position

        unvisited_nodes.remove(best_node)
        current_state.instance.add_edge(best_position[0], best_node)
        current_state.dgl_graph.edata['e_feat'][edge_ends_to_edge_index(best_position[0],
                                                                   best_node,
                                                                   number_of_nodes)][1] = 1
        current_state.instance.add_edge(best_node, best_position[1])
        current_state.dgl_graph.edata['e_feat'][edge_ends_to_edge_index(best_node,
                                                                   best_position[1],
                                                                   number_of_nodes)][1] = 1
        if best_position[0] != best_position[1]:
            current_state.instance.remove_edge(best_position[0], best_position[1])
            current_state.dgl_graph.edata['e_feat'][edge_ends_to_edge_index(best_position[0],
                                                                       best_position[1],
                                                                       number_of_nodes)][1] = 0

    return current_state
