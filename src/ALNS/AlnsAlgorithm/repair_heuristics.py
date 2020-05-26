from src.ALNS.AlnsAlgorithm.compute_distances import compute_minimum_cost_position


def greedy_insertion(current_state, random_state):
    inserted = current_state.copy()

    unvisited_nodes = [node for node in inserted.instance.nodes() if inserted.instance.degree(node) == 0]

    while len(unvisited_nodes) > 0:
        best_node = unvisited_nodes[0]
        best_node_cost, best_position = compute_minimum_cost_position(inserted, best_node)
        for node in unvisited_nodes[1:]:
            node_cost, position = compute_minimum_cost_position(inserted, node)
            if node_cost < best_node_cost:
                best_node = node
                best_node_cost = node_cost
                best_position = position

        unvisited_nodes.remove(best_node)
        inserted.instance.add_edge(best_position[0], best_node)
        inserted.instance.add_edge(best_node, best_position[1])
        if best_position[0] != best_position[1]:
            inserted.instance.remove_edge(best_position[0], best_position[1])

    return inserted
