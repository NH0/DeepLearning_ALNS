import torch

import torch.nn as nn
import NeuralNetwork.parameters as parameters

EPSILON = parameters.EPSILON


class GatedGCNLayer(nn.Module):
    """
    Defines a layer of a Gated Graph Convolutional Network based on https://arxiv.org/pdf/1711.07553v2.pdf.

    The equation defining the message passing is the following :
    h = hi + ReLU(BN(U x hi + Sum(eta_ij * V x hj)))
    where :
    eta_ij = sigmoid(e_ij) / (Sum(sigmoid(e_ij)) + epsilon)
    and :
    e_ij = e_ij + ReLU(BN(W1 x e_ij + W2 x hi + W3 x hj))

    BN is a Batch normalization.

    In order to have different feature sizes between layers, we add a linear transformation
    for nodes to hi :
    h = embedding_node x hi + ReLU(BN(W1 x hi + Sum(eta_ij * W2 x hj)))
    for edges to e_ij :
    e_ij = embedding_edge x e_ij + ReLU(BN(W1 x e_ij + W2 x hi + W3 x hj))


    And to allow different sizes between node features and edge features, we also add a linear transformation to eta_ij
    h = embedding_node x hi + ReLU(BN(W1 x hi + Sum(embedding_eta x eta_ij * W2 x hj)))
    """

    def __init__(self, input_node_features, output_node_features,
                 input_edge_features, output_edge_features,
                 dropout_probability, has_dropout=False):
        super(GatedGCNLayer, self).__init__()

        self.input_node_features = input_node_features
        self.output_node_features = output_node_features
        self.input_edge_features = input_edge_features
        self.output_edge_features = output_edge_features

        # This embeddings are used to change the dimension of the node and edge features from one layer to another
        # Otherwise each layer would have the same number of features
        self.embedding_node = nn.Linear(input_node_features, output_node_features, bias=False)
        self.embedding_edge = nn.Linear(input_edge_features, output_edge_features, bias=False)

        # This embedding is used to be able to add the edge feature vector to the node feature vector
        self.embedding_eta = nn.Linear(output_edge_features, output_node_features, bias=False)

        self.has_dropout = has_dropout
        self.dropout_probability = dropout_probability

        self.U = nn.Linear(input_node_features, output_node_features, bias=True)
        self.V = nn.Linear(input_node_features, output_node_features, bias=True)

        self.W1 = nn.Linear(input_edge_features, output_edge_features, bias=True)
        self.W2 = nn.Linear(input_node_features, output_edge_features, bias=True)
        self.W3 = nn.Linear(input_node_features, output_edge_features, bias=True)

        self.activation = nn.ReLU()
        self.h_BN = nn.BatchNorm1d(output_node_features)
        self.e_BN = nn.BatchNorm1d(output_edge_features)
        self.sigmoid = nn.Sigmoid()

    def message_function(self, edges):
        Vh_j = edges.src['Vh']
        e_ij = edges.data['e'] + self.activation(self.e_BN(edges.data['W1e'] + edges.src['W2h'] + edges.dst['W3h']))
        edges.data['e'] = e_ij

        return {'Vh_j': Vh_j, 'e_ij': e_ij}

    def reduce_function(self, nodes):
        Uh_i = nodes.data['Uh']
        Vh_j = nodes.mailbox['Vh_j']

        e = nodes.mailbox['e_ij']
        sigma_ij = self.embedding_eta(self.sigmoid(e))

        h = nodes.data['h'] + self.activation(self.h_BN(Uh_i + torch.sum(sigma_ij * Vh_j, dim=1)
                                                        / (torch.sum(sigma_ij, dim=1) + EPSILON)))

        return {'h': h}

    def forward(self, graph, h, e):
        graph.ndata['h'] = self.embedding_node(h)
        graph.ndata['Uh'] = self.U(h)
        graph.ndata['Vh'] = self.V(h)
        graph.ndata['W2h'] = self.W2(h)
        graph.ndata['W3h'] = self.W3(h)

        graph.edata['e'] = self.embedding_edge(e)
        graph.edata['W1e'] = self.W1(e)

        graph.update_all(message_func=self.message_function, reduce_func=self.reduce_function)
        h = graph.ndata['h']
        e = graph.edata['e']

        if self.has_dropout:
            h = torch.nn.functional.dropout(h, self.dropout_probability)
            e = torch.nn.functional.dropout(e, self.dropout_probability)

        return h, e


class GCN(nn.Module):
    """
    Classifies an alns iteration for the CVRP (destruction & reconstruction).

    The network predicts whether the iteration will improve, worsen or keep the total cost of the CVRP solution.
    The network is based on Gated Graph Convolution layers followed by a Fully connected layer with and output of size
    3 (for the 3 possible classes).
    """

    def __init__(self,
                 input_node_features, hidden_node_dimension_list,
                 input_edge_features, hidden_edge_dimension_list,
                 hidden_linear_dimension,
                 output_feature,
                 dropout_probability,
                 device):
        super(GCN, self).__init__()

        if len(hidden_node_dimension_list) != len(hidden_edge_dimension_list):
            print("Node dimensions and edge dimensions lists aren't the same size !\nExiting...")
            exit(1)

        self.convolutions = [GatedGCNLayer(input_node_features, hidden_node_dimension_list[0],
                                           input_edge_features, hidden_edge_dimension_list[0],
                                           dropout_probability).to(device)]
        self.add_module('convolution1', self.convolutions[0])

        for i in range(1, len(hidden_node_dimension_list)):
            self.convolutions.append(GatedGCNLayer(hidden_node_dimension_list[i - 1], hidden_node_dimension_list[i],
                                                   hidden_edge_dimension_list[i - 1], hidden_edge_dimension_list[i],
                                                   dropout_probability).to(device))
            self.add_module('convolution' + str(i + 1), self.convolutions[-1])

        self.linear1 = nn.Linear(hidden_node_dimension_list[-1], hidden_linear_dimension)
        self.linear2 = nn.Linear(hidden_linear_dimension, hidden_linear_dimension)
        self.linear3 = nn.Linear(hidden_linear_dimension, hidden_linear_dimension)
        self.linear4 = nn.Linear(hidden_linear_dimension, output_feature)

    def forward(self, graph, h, e):
        for convolution in self.convolutions:
            h, e = convolution(graph, h, e)

        # Return a tensor of shape (hidden_dimension)
        h = torch.mean(h, dim=0)
        h = self.linear1(h)
        h = self.linear2(h)
        h = self.linear3(h)
        h = self.linear4(h)

        return h
