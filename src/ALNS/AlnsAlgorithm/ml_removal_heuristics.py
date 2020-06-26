import numpy as np
from torch import argmax
from torch import device as torch_device
from torch.cuda import is_available as is_cuda_available
from torch import load as torch_load
from torch import no_grad

from src.ALNS.AlnsAlgorithm.removal_heuristics import compute_number_of_clients_to_remove, remove_nodes
from src.ALNS.CVRP.from_nx_to_dgl import edge_ends_to_edge_index
import src.ALNS.settings as settings
import src.NeuralNetwork.parameters as parameters

from src.NeuralNetwork.GCN.GCN_net import GCNNet
from src.NeuralNetwork.Gated_GCN.gated_gcn_net import GatedGCNNet

NETWORK_GCN = parameters.NETWORK_GCN
NETWORK_GATEDGCN = parameters.NETWORK_GATEDGCN

MODEL_PARAMETERS_PATH = parameters.MODEL_PARAMETERS_PATH

HIDDEN_NODE_DIMENSIONS = parameters.HIDDEN_NODE_DIMENSIONS
HIDDEN_EDGE_DIMENSIONS = parameters.HIDDEN_EDGE_DIMENSIONS
HIDDEN_LINEAR_DIMENSIONS = parameters.HIDDEN_LINEAR_DIMENSIONS
OUTPUT_SIZE = parameters.OUTPUT_SIZE

ACCEPT_NODES_RANDOMLY = settings.ACCEPT_NODES_RANDOMLY


def select_random_nodes(state, random_state):
    # We create a list containing the indexes of the client nodes
    # Their indexes start after their indexes of the depots by construction
    list_of_nodes = np.arange(state.size) + state.number_of_depots

    return random_state.choice(list_of_nodes, compute_number_of_clients_to_remove(state), replace=False)


def update_features_from_destroyed_nodes(state, nodes_to_destroy):
    number_of_nodes = state.instance.number_of_nodes()
    for node in nodes_to_destroy:
        next_node = next(state.instance.neighbors(node))
        previous_node = next(state.instance.predecessors(node))
        state.dgl_graph.ndata['n_feat'][node][2] = 1
        state.dgl_graph.edata['e_feat'][edge_ends_to_edge_index(previous_node, node, number_of_nodes)][1] = 0
        state.dgl_graph.edata['e_feat'][edge_ends_to_edge_index(node, next_node, number_of_nodes)][1] = 0
        state.dgl_graph.edata['e_feat'][edge_ends_to_edge_index(previous_node, next_node, number_of_nodes)][1] = 1


def define_ml_removal_heuristic(number_of_node_features, number_of_edge_features,
                                network_params_file,
                                network=NETWORK_GCN):
    device = 'cuda' if is_cuda_available() else 'cpu'
    dropout = 0.0
    training_state = torch_load(MODEL_PARAMETERS_PATH + network_params_file, map_location=torch_device(device))

    if network == NETWORK_GATEDGCN:
        net_params = {
            'in_dim': number_of_node_features,
            'in_dim_edge': number_of_edge_features,
            'hidden_dim': 70,
            'out_dim': 70,
            'n_classes': OUTPUT_SIZE,
            'dropout': dropout,
            'L': len(HIDDEN_NODE_DIMENSIONS),
            'readout': 'mean',
            'graph_norm': False,
            'batch_norm': False,
            'residual': False,
            'edge_feat': True,
            'device': device
        }
        model = GatedGCNNet(net_params)
        model.load_state_dict(training_state)
    else:
        model = GCNNet(input_node_features=number_of_node_features,
                       hidden_node_dimension_list=HIDDEN_NODE_DIMENSIONS,
                       input_edge_features=number_of_edge_features,
                       hidden_edge_dimension_list=HIDDEN_EDGE_DIMENSIONS,
                       hidden_linear_dimension_list=HIDDEN_LINEAR_DIMENSIONS,
                       output_feature=OUTPUT_SIZE,
                       dropout_probability=dropout,
                       device=device)
        model.load_state_dict(training_state['graph_convolutional_network_state'])

    model.eval()

    @no_grad()
    def ml_removal_heuristic(state, random_state):
        nodes_to_destroy = select_random_nodes(state, random_state)
        update_features_from_destroyed_nodes(state, nodes_to_destroy)
        prediction = argmax(model(state.dgl_graph,
                                  state.dgl_graph.ndata['n_feat'],
                                  state.dgl_graph.edata['e_feat'],
                                  state.node_snorm,
                                  state.edge_snorm), dim=1)
        are_nodes_accepted = (prediction == 2)
        number_of_tries = state.size
        while not are_nodes_accepted and number_of_tries > 0:
            nodes_to_destroy = select_random_nodes(state, random_state)
            update_features_from_destroyed_nodes(state, nodes_to_destroy)
            prediction = argmax(model(state.dgl_graph,
                                      state.dgl_graph.ndata['n_feat'],
                                      state.dgl_graph.edata['e_feat'],
                                      state.node_snorm,
                                      state.edge_snorm), dim=1).item()
            random_accept_nodes = True if np.random.random() < ACCEPT_NODES_RANDOMLY else False
            are_nodes_accepted = True if (prediction == 2 or random_accept_nodes) else False
            number_of_tries -= 1

        return remove_nodes(state, list(nodes_to_destroy))

    return ml_removal_heuristic
