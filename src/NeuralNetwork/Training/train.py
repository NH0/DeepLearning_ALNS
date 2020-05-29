import torch
import pickle
import datetime

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import src.NeuralNetwork.parameters as parameters

from src.NeuralNetwork.Dataset.dataset import create_dataset_from_statistics, pickle_dataset, unpickle_dataset
from src.NeuralNetwork.GCN import GCN

MODEL_PARAMETERS_PATH = parameters.MODEL_PARAMETERS_PATH
DATASET_PREFIX = parameters.DATASET_PREFIX
ALNS_STATISTICS_FILE = parameters.ALNS_STATISTICS_FILE
DATASET_NAME = parameters.DATASET_NAME


HIDDEN_NODE_DIMENSIONS = parameters.HIDDEN_NODE_DIMENSIONS
HIDDEN_EDGE_DIMENSIONS = parameters.HIDDEN_EDGE_DIMENSIONS
HIDDEN_LINEAR_DIMENSIONS = parameters.HIDDEN_LINEAR_DIMENSIONS
OUTPUT_SIZE = parameters.OUTPUT_SIZE
DROPOUT_PROBABILITY = parameters.DROPOUT_PROBABILITY
MAX_EPOCH = parameters.MAX_EPOCH
EPSILON = parameters.EPSILON

INITIAL_LEARNING_RATE = parameters.INITIAL_LEARNING_RATE
LEARNING_RATE_DECREASE_FACTOR = parameters.LEARNING_RATE_DECREASE_FACTOR

DISPLAY_EVERY_N_EPOCH = parameters.DISPLAY_EVERY_N_EPOCH


def evaluate(network, inputs_test, labels, train_mask):
    """
    Evaluate a neural network on a given test set.

    Parameters
    ----------
    network : the network to evaluate
    inputs_test : the test dataset, containing DGL graphs
    labels : the expected values to be returned by the network
    train_mask : the inverse of mask to apply on the labels to keep only the labels corresponding to the test set

    Returns
    -------
    The proportion of right predictions
    """
    # Inverse the mask to have the test mask
    test_mask = ~train_mask
    network.eval()
    with torch.no_grad():
        correct = 0
        for index, graph in enumerate(inputs_test):
            logits = network(graph, graph.ndata['n_feat'], graph.edata['e_feat'])
            logp = F.softmax(logits, dim=0)
            predicted_class = torch.argmax(logp, dim=0).item()
            true_class = torch.argmax(labels[test_mask][index], dim=0).item()
            correct += predicted_class == true_class

    return correct / len(inputs_test)


def evaluate_random(labels, train_mask, number_of_test_values):
    test_mask = ~train_mask
    correct = 0
    for i in range(number_of_test_values):
        true_class = torch.argmax(labels[test_mask][i], dim=0).item()
        correct += np.random.randint(0, 3) == true_class

    return correct / number_of_test_values


def evaluate_with_null_iteration(labels, train_mask, number_of_test_values):
    test_mask = ~train_mask
    correct = 0
    for i in range(number_of_test_values):
        true_class = torch.argmax(labels[test_mask][i], dim=0).item()
        correct += 1 == true_class

    return correct / number_of_test_values


def display_proportion_of_null_iterations(train_mask, labels, training_set_size, device):
    number_of_iterations = len(train_mask)
    number_of_total_null_iterations = 0
    number_of_train_null_iterations = 0
    null_label = torch.tensor([0, 1, 0], dtype=torch.float, device=device)
    for index, iteration in enumerate(labels):
        if torch.equal(iteration, null_label):
            number_of_total_null_iterations += 1
            if train_mask[index] == 1:
                number_of_train_null_iterations += 1
    print("{:.2%} of total null iterations".format(
        round(number_of_total_null_iterations / number_of_iterations, 4)
    ))
    print("{:.2%} of null iterations in training set".format(
        round(number_of_train_null_iterations / training_set_size, 4)
    ))
    print("Dataset size : {}".format(number_of_iterations))
    print("Training set size : {}".format(training_set_size))


def save_model_parameters(graph_convolutional_network,
                          optimizer,
                          hidden_node_dimensions, hidden_edge_dimensions, hidden_linear_dimensions,
                          initial_learning_rate,
                          epoch,
                          training_loss,
                          device):
    name_model_parameters_file = 'GCNparams_ep' + str(epoch) + '_ndim'
    for dim in hidden_node_dimensions:
        name_model_parameters_file += str(dim) + '.'
    name_model_parameters_file += '_edim'
    for dim in hidden_edge_dimensions:
        name_model_parameters_file += str(dim) + '.'
    name_model_parameters_file += '_ldim'
    for dim in hidden_linear_dimensions:
        name_model_parameters_file += str(dim) + '.'
    name_model_parameters_file += '_lr' + str(initial_learning_rate)
    name_model_parameters_file += '_dev' + device
    name_model_parameters_file += '.pt'
    torch.save({'graph_convolutional_network_state': graph_convolutional_network.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'epoch': epoch,
                'training_loss': training_loss},
               MODEL_PARAMETERS_PATH + name_model_parameters_file)
    print("Successfully saved the model's parameters in {}".format(MODEL_PARAMETERS_PATH + name_model_parameters_file))


def main(recreate_dataset=False,
         hidden_node_dimensions=None,
         hidden_edge_dimensions=None,
         hidden_linear_dimensions=HIDDEN_LINEAR_DIMENSIONS,
         output_size=OUTPUT_SIZE,
         dropout_probability=DROPOUT_PROBABILITY,
         max_epoch=MAX_EPOCH, epsilon=EPSILON,
         initial_learning_rate=INITIAL_LEARNING_RATE,
         learning_rate_decrease_factor=LEARNING_RATE_DECREASE_FACTOR,
         save_parameters_on_exit=True,
         load_parameters_from_file=None,
         **keywords_args):
    # Avoid mutable default arguments
    if hidden_edge_dimensions is None:
        hidden_edge_dimensions = HIDDEN_EDGE_DIMENSIONS
    if hidden_node_dimensions is None:
        hidden_node_dimensions = HIDDEN_NODE_DIMENSIONS

    """
    Use GPU if available.
    """
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    print("#" * 50)
    print("# Date : {0:%y}-{0:%m}-{0:%d}_{0:%H}-{0:%M}".format(datetime.datetime.now()))
    print("# Hidden node dimensions : {}".format(hidden_node_dimensions))
    print("# Hidden edge dimensions : {}".format(hidden_edge_dimensions))
    print("# Hidden linear dimensions : {}".format(hidden_linear_dimensions))
    print("# Dropout probability : {}".format(dropout_probability))
    print("# Max epoch : {}".format(max_epoch))
    print("# Initial learning rate : {}".format(initial_learning_rate))
    print("# Device : {}".format(device))
    print("#" * 50)

    if recreate_dataset:
        print("Creating dataset from ALNS statistics :")
        if 'alns_statistics_file' not in keywords_args:
            alns_statistics_file = ALNS_STATISTICS_FILE
        else:
            alns_statistics_file = keywords_args['alns_statistics_file']
        """
        Create the train and test sets.
        """
        inputs_train, inputs_test, train_mask, labels = create_dataset_from_statistics(alns_statistics_file,
                                                                                       device,
                                                                                       epsilon)
        print("Created dataset !")
        if 'pickle_dataset' in keywords_args:
            if keywords_args['pickle_dataset']:
                dataset_filename = DATASET_PREFIX + alns_statistics_file
                pickle_dataset(dataset_filename, inputs_train, inputs_test, train_mask, labels)
    else:
        print("Retrieving dataset ... ", end='', flush=True)
        if 'dataset_name' not in keywords_args:
            dataset_name = DATASET_NAME
        else:
            dataset_name = keywords_args['dataset_name']
        inputs_train, inputs_test, train_mask, labels = unpickle_dataset(dataset_name)
        print("Done !", flush=True)

    number_of_node_features = len(inputs_test[0].ndata['n_feat'][0])
    number_of_edge_features = len(inputs_test[0].edata['e_feat'][0])

    """
    Create the gated graph convolutional network
    """
    graph_convolutional_network = GCN(input_node_features=number_of_node_features,
                                      hidden_node_dimension_list=hidden_node_dimensions,
                                      input_edge_features=number_of_edge_features,
                                      hidden_edge_dimension_list=hidden_edge_dimensions,
                                      hidden_linear_dimension_list=hidden_linear_dimensions,
                                      output_feature=output_size,
                                      dropout_probability=dropout_probability,
                                      device=device)
    graph_convolutional_network = graph_convolutional_network.to(device)
    print("Created GCN", flush=True)

    """
    Define the optimizer, the learning rate scheduler and the loss function.
    We use the Adam optimizer and a MSE loss.
    """
    optimizer = torch.optim.Adam(graph_convolutional_network.parameters(), lr=initial_learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=learning_rate_decrease_factor)
    loss_function = nn.MSELoss()

    """
    Resume training state
    """
    initial_epoch = 0
    training_loss = []
    if load_parameters_from_file is not None:
        try:
            training_state = torch.load(MODEL_PARAMETERS_PATH + load_parameters_from_file)
            graph_convolutional_network.load_state_dict(training_state['graph_convolutional_network_state'])
            graph_convolutional_network.train()
            optimizer.load_state_dict(training_state['optimizer_state'])
            initial_epoch = training_state['epoch']
            training_loss = training_state['training_loss']
            print("Loaded parameters values from {}".format(MODEL_PARAMETERS_PATH + load_parameters_from_file))
            print("Resuming at epoch {}".format(initial_epoch))
        except (pickle.UnpicklingError, TypeError, RuntimeError, KeyError) as exception_value:
            print("Unable to load parameters from {}".format(MODEL_PARAMETERS_PATH + load_parameters_from_file))
            print("Exception : {}".format(exception_value))
            should_continue = ''
            while should_continue != 'y' or should_continue != 'n':
                should_continue = input("Continue anyway with random parameters ? (y/n) ")
            if should_continue == 'n':
                exit(1)

    """
    Display the proportion of null iterations (iterations that do not change the cost value of the CVRP solution.
    """
    display_proportion_of_null_iterations(train_mask, labels, len(inputs_train), device)

    print("\nStarting training {}\n".format(chr(8987)))

    """
    Train the network.
    """
    for epoch in range(initial_epoch, max_epoch + 1):
        try:
            running_loss = 0.0
            if epoch % DISPLAY_EVERY_N_EPOCH == 1:
                accuracy = evaluate(graph_convolutional_network, inputs_test, labels, train_mask)
                random_accuracy = evaluate_random(labels, train_mask, len(inputs_test))
                guessing_null_iteration_accuracy = evaluate_with_null_iteration(labels, train_mask, len(inputs_test))
                print("Epoch {:d}, loss {:.6f}, accuracy {:.4f}, random accuracy {:.4f}, "
                      "always guessing null iterations {:.4f}"
                      .format(epoch, training_loss[epoch - 1], accuracy, random_accuracy,
                              guessing_null_iteration_accuracy))

            for index, graph in enumerate(inputs_train):
                logits = graph_convolutional_network(graph, graph.ndata['n_feat'], graph.edata['e_feat'])
                logp = F.softmax(logits, dim=0)
                loss = loss_function(logp, labels[train_mask][index])

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step(loss)

                running_loss += loss
            training_loss.append(running_loss / len(inputs_train))

        except KeyboardInterrupt:
            print("Received keyboard interrupt.")
            if save_parameters_on_exit:
                print("Saving parameters before quiting ...", flush=True)
                save_model_parameters(graph_convolutional_network,
                                      optimizer,
                                      hidden_node_dimensions, hidden_edge_dimensions, hidden_linear_dimensions,
                                      initial_learning_rate, epoch, training_loss, device)
            exit(0)

    if save_parameters_on_exit:
        save_model_parameters(graph_convolutional_network,
                              optimizer,
                              hidden_node_dimensions, hidden_edge_dimensions, hidden_linear_dimensions,
                              initial_learning_rate, max_epoch, training_loss, device)


if __name__ == '__main__':
    main(dataset_name=
         'inputs_mask_labels_dataset_50-50_1inst_50nod_40cap_1dep_1000iter_0.8decay_0.35destr_18determ.pickle',
         max_epoch=20)
