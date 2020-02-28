degree_of_destruction = 0.2

def number_of_edges_to_remove(state):
    return int(state.instance.number_of_edges() * degree_of_destruction)

def random_removal(current_state, random_state):
    destroyed = current_state.copy()

    # We choose nodes and destroy the existing edge
    nodes_to_destroy = random_state.choice(destroyed.instance.number_of_nodes(), numberOfEdgesToRemove(current_state), replace=False)
    for node in nodes_to_destroy:
        detroyed.instance.remove_edge(node, next(iter(destroyed.instance.adj[node])))

    return destroyed
