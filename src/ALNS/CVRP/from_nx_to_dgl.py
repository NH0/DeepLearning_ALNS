import dgl
import torch


def edge_ends_to_edge_index(u, v, number_of_nodes):
    if u == v:
        raise ValueError("Edge with same origin and destination {}".format(u))
    return u * (number_of_nodes - 1) + v if v < u else u * (number_of_nodes - 1) + v - 1


def make_complete_nx_graph(nx_graph):
    number_of_nodes = nx_graph.number_of_nodes()
    # Create a list containing all possible edges
    edges_in_complete_graph = [(u, v) for u in range(number_of_nodes) for v in range(number_of_nodes) if u != v]
    for u, v in edges_in_complete_graph:
        # Networkx will not add the edge if it already exists
        nx_graph.add_edge(u, v)


def generate_dgl_graph(nx_graph):
    """
    Convert a networkx graph to a DGL graph.

    Parameters
    ----------
    nx_graph : a networkx graph

    Returns
    -------
    dgl_graph the nx_graph converted to DGL
    """
    dgl_graph = dgl.DGLGraph()
    dgl_graph.from_networkx(nx_graph=nx_graph)
    dgl_graph.set_n_initializer(dgl.init.zero_initializer)

    return dgl_graph


def initialize_dgl_features(cvrp_state, dgl_graph, destroyed_nodes, list_of_edges, device):
    nx_graph = cvrp_state.instance
    number_of_nodes = dgl_graph.number_of_nodes()
    number_of_edges = dgl_graph.number_of_edges()

    node_features = [[cvrp_state.capacity - nx_graph.nodes[node]['demand'],
                      1 if nx_graph.nodes[node]['isDepot'] else 0,
                      1 if node in destroyed_nodes else 0]
                     for node in range(number_of_nodes)]
    edge_features = [[cvrp_state.distances[u][v],
                      1 if (u, v) in list_of_edges else 0]
                     for u in range(number_of_nodes) for v in range(number_of_nodes) if u != v]

    node_features_tensor = torch.tensor(node_features, dtype=torch.float, device=device)
    edge_features_tensor = torch.tensor(edge_features, dtype=torch.float, device=device)
    dgl_graph.ndata['n_feat'] = node_features_tensor
    dgl_graph.edata['e_feat'] = edge_features_tensor

    node_snorm = torch.tensor((number_of_nodes, 1),
                              dtype=torch.float, device=device).fill_(1./float(number_of_nodes)).sqrt()
    edge_snorm = torch.tensor((number_of_edges, 1),
                              dtype=torch.float, device=device).fill_(1./float(number_of_edges)).sqrt()

    return node_snorm, edge_snorm
