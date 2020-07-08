import torch
import pickle
import datetime

import torch.nn as nn
import src.NeuralNetwork.parameters as parameters

from torch.utils.data import DataLoader
from src.NeuralNetwork.Dataset.dataset_utils import create_dataset, collate, generate_all_inputs_and_labels
from src.NeuralNetwork.GCN.GCN_net import GCNNet
from src.NeuralNetwork.Gated_GCN.gated_gcn_net import GatedGCNNet

MODEL_PARAMETERS_PATH = parameters.MODEL_PARAMETERS_PATH
INPUTS_LABELS_PATH = parameters.INPUTS_LABELS_PATH
INPUTS_LABELS_PREFIX = parameters.INPUTS_LABELS_PREFIX
ALNS_STATISTICS_FILE = parameters.ALNS_STATISTICS_FILE
INPUTS_LABELS_NAME = parameters.INPUTS_LABELS_NAME

HIDDEN_NODE_DIMENSIONS = parameters.HIDDEN_NODE_DIMENSIONS
HIDDEN_EDGE_DIMENSIONS = parameters.HIDDEN_EDGE_DIMENSIONS
HIDDEN_LINEAR_DIMENSIONS = parameters.HIDDEN_LINEAR_DIMENSIONS
OUTPUT_SIZE = parameters.OUTPUT_SIZE
DROPOUT_PROBABILITY = parameters.DROPOUT_PROBABILITY
MAX_EPOCH = parameters.MAX_EPOCH
BATCH_SIZE = parameters.BATCH_SIZE

INITIAL_LEARNING_RATE = parameters.INITIAL_LEARNING_RATE
MIN_LEARNING_RATE = parameters.MIN_LEARNING_RATE
LEARNING_RATE_DECREASE_FACTOR = parameters.LEARNING_RATE_DECREASE_FACTOR
PATIENCE = parameters.PATIENCE

BETA = parameters.BETA

DISPLAY_EVERY_N_EPOCH = parameters.DISPLAY_EVERY_N_EPOCH

NETWORK_GCN = parameters.NETWORK_GCN
NETWORK_GATEDGCN = parameters.NETWORK_GATEDGCN


def make_training_step(graph_convolutional_network, loss_function, softmax_function, optimizer):
    def train_step(graph_batch, label_batch):
        logits = graph_convolutional_network(graph_batch, graph_batch.ndata['n_feat'], graph_batch.edata['e_feat'],
                                             0, 0)
        loss = loss_function(logits, label_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.detach().item()

    return train_step


def evaluate(network, loss_function, softmax_function, test_loader, test_set_size):
    """
    Evaluate a neural network on a given test set.

    Parameters
    ----------
    softmax_function
    loss_function
    test_loader : the test dataset
    network : the network to evaluate

    Returns
    -------
    The proportion of right predictions
    """
    running_loss = 0.0
    confusion_matrix = {  # Of shape [predicted value][real value]
        0: {0: 0, 1: 0, 2: 0},
        1: {0: 0, 1: 0, 2: 0},
        2: {0: 0, 1: 0, 2: 0},
    }
    batch_size = -1
    network.eval()
    with torch.no_grad():
        correct = 0
        for graph_batch, label_batch in test_loader:
            if batch_size == -1:
                batch_size = label_batch.size(0)
            logits = network(graph_batch, graph_batch.ndata['n_feat'], graph_batch.edata['e_feat'], 0, 0)
            running_loss += loss_function(logits, label_batch).detach().item()
            predicted_classes = torch.argmax(logits, dim=1).detach()
            correct += (predicted_classes == label_batch).sum().item()
            for predicted_class, label in zip(predicted_classes, label_batch):
                confusion_matrix[predicted_class.item()][label.item()] += 1

    if batch_size <= 0:
        print("Error : batch size is {}".format(batch_size))
        exit(1)

    return correct / test_set_size, running_loss / len(test_loader), confusion_matrix


def evaluate_random(test_loader, test_set_size):
    correct = 0
    batch_size = -1
    for _, label_batch in test_loader:
        if batch_size == -1:
            batch_size = label_batch.size(0)
        random_tensor = torch.randint(0, OUTPUT_SIZE, size=label_batch.size(), device=label_batch.device)
        correct += (random_tensor == label_batch).sum().item()

    return correct / test_set_size


def evaluate_with_null_iteration(test_loader, test_set_size):
    correct = 0
    batch_size = -1
    for _, label_batch in test_loader:
        if batch_size == -1:
            batch_size = label_batch.size(0)
        ones_tensor = torch.ones(size=label_batch.size(), device=label_batch.device)
        correct += (ones_tensor == label_batch).sum().item()

    return correct / test_set_size


def compute_classes_weights(train_loader, test_loader, device, beta):
    training_set_size = 0
    test_set_size = 0
    number_of_elements_train_set = {
        0: 0, 1: 0, 2: 0
    }
    number_of_elements_test_set = {
        0: 0, 1: 0, 2: 0
    }
    for _, labels in train_loader:
        for label in labels:
            number_of_elements_train_set[label.item()] += 1
            training_set_size += 1
    for _, labels in test_loader:
        for label in labels:
            number_of_elements_test_set[label.item()] += 1
            test_set_size += 1
    print("{:<20}{:^7.2%}{:^7.2%}{:^7.2%}".format(
        'Training set',
        round(number_of_elements_train_set[0] / training_set_size, 4),
        round(number_of_elements_train_set[1] / training_set_size, 4),
        round(number_of_elements_train_set[2] / training_set_size, 4),
    ))
    print("{:<20}{:^7.2%}{:^7.2%}{:^7.2%}".format(
        'Test set',
        round(number_of_elements_test_set[0] / test_set_size, 4),
        round(number_of_elements_test_set[1] / test_set_size, 4),
        round(number_of_elements_test_set[2] / test_set_size, 4),
    ))
    print("Dataset size : {}".format(training_set_size + test_set_size))
    print("Training set size : {}".format(training_set_size))
    # train_weights = [
    #     1 - number_of_elements_train_set[0] / training_set_size,
    #     1 - number_of_elements_train_set[1] / training_set_size,
    #     1 - number_of_elements_train_set[2] / training_set_size,
    # ]
    # train_weights = [
    #     1 / number_of_elements_train_set[0],
    #     1 / number_of_elements_train_set[1],
    #     1 / number_of_elements_train_set[2],
    # ]
    effective_train_weights = []
    denominator = 0
    for i in range(3):
        temp = 1
        for j in range(3):
            if j != i:
                temp *= 1 - beta ** number_of_elements_train_set[j]
        denominator += temp
    for i in range(3):
        numerator = 1
        for j in range(3):
            if j != i:
                numerator *= 1 - beta ** number_of_elements_train_set[j]
        effective_train_weights.append(numerator / denominator)
    print("{:<20}{:^7.2}{:^7.2}{:^7.2}".format(
        'Training weights',
        round(effective_train_weights[0], 4),
        round(effective_train_weights[1], 4),
        round(effective_train_weights[2], 4),
    ))

    return torch.tensor(effective_train_weights, device=device), training_set_size, test_set_size


def display_confusion_matrix(confusion_matrix):
    print("Confusion matrix :")
    print("{:^20}|{:^7}|{:^7}|{:^7}".format('Predicted \\ Real', '0', '1', '2'))
    print("-" * 38)
    print("{1:^20}|{0[0]:^7}|{0[1]:^7}|{0[2]:^7}".format(confusion_matrix[0], '0 (delta > 0)'))
    print("-" * 38)
    print("{1:^20}|{0[0]:^7}|{0[1]:^7}|{0[2]:^7}".format(confusion_matrix[1], '1 (delta = 0)'))
    print("-" * 38)
    print("{1:^20}|{0[0]:^7}|{0[1]:^7}|{0[2]:^7}".format(confusion_matrix[2], '2 (delta < 0)'))


def save_model_parameters(graph_convolutional_network,
                          optimizer,
                          softmax_function_name,
                          initial_learning_rate,
                          epoch,
                          training_loss, test_loss,
                          device):
    name_model_parameters_file = 'GCNparams_ep' + str(epoch)
    name_model_parameters_file += '_lr' + str(initial_learning_rate)
    name_model_parameters_file += '_dev' + device
    name_model_parameters_file += '_' + softmax_function_name
    name_model_parameters_file += '.pt'
    torch.save({'graph_convolutional_network_state': graph_convolutional_network.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'epoch': epoch,
                'training_loss': training_loss,
                'test_loss': test_loss},
               MODEL_PARAMETERS_PATH + name_model_parameters_file)
    print("Successfully saved the model's parameters in {}".format(MODEL_PARAMETERS_PATH + name_model_parameters_file))


def main(recreate_dataset=False,
         batch_size=BATCH_SIZE,
         test_batch_size=BATCH_SIZE,
         weight_loss=False,
         network=NETWORK_GCN,
         hidden_node_dimensions=None,
         hidden_edge_dimensions=None,
         hidden_linear_dimensions=None,
         output_size=OUTPUT_SIZE,
         dropout_probability=DROPOUT_PROBABILITY,
         max_epoch=MAX_EPOCH,
         initial_learning_rate=INITIAL_LEARNING_RATE,
         min_learning_rate=MIN_LEARNING_RATE,
         learning_rate_decrease_factor=LEARNING_RATE_DECREASE_FACTOR,
         patience=PATIENCE,
         save_parameters_on_exit=True,
         load_parameters_from_file=None,
         **keywords_args):
    # Avoid mutable default arguments
    if hidden_edge_dimensions is None:
        hidden_edge_dimensions = HIDDEN_EDGE_DIMENSIONS
    if hidden_node_dimensions is None:
        hidden_node_dimensions = HIDDEN_NODE_DIMENSIONS
    if hidden_linear_dimensions is None:
        hidden_linear_dimensions = HIDDEN_LINEAR_DIMENSIONS

    """
    Use GPU if available.
    """
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    if recreate_dataset:
        print("Creating dataset from ALNS statistics :")
        if 'alns_statistics_file' not in keywords_args:
            alns_statistics_file = ALNS_STATISTICS_FILE
        else:
            alns_statistics_file = keywords_args['alns_statistics_file']
        """
        Create the train and test sets.
        """
        inputs, labels = generate_all_inputs_and_labels(alns_statistics_file, device)
        print("Created dataset !")
        if 'pickle_dataset' in keywords_args and type(keywords_args['pickle_dataset']) is bool:
            if keywords_args['pickle_dataset']:
                inputs_labels_name = INPUTS_LABELS_PREFIX + alns_statistics_file
                # Cannot use torch.save on DGL graphs, see https://github.com/dmlc/dgl/issues/1524
                # Using pickle.dump instead
                with open(INPUTS_LABELS_PATH + inputs_labels_name, 'wb') as dataset_file:
                    pickle.dump({
                        'inputs': inputs,
                        'labels': labels
                    }, dataset_file)
                print("Successfully saved the data in {}".format(INPUTS_LABELS_PATH + inputs_labels_name))
    else:
        if 'inputs_labels_name' not in keywords_args:
            inputs_labels_name = INPUTS_LABELS_NAME
        else:
            inputs_labels_name = keywords_args['inputs_labels_name']
        print("Retrieving dataset {} ... ".format(inputs_labels_name), end='', flush=True)
        with open(INPUTS_LABELS_PATH + inputs_labels_name, 'rb') as dataset_file:
            dataset = pickle.load(dataset_file)
        inputs = dataset['inputs']
        labels = dataset['labels']
        print("Done !", flush=True)

    train_set, test_set = create_dataset(inputs, labels)
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, collate_fn=collate)
    test_loader = DataLoader(dataset=test_set, batch_size=test_batch_size, collate_fn=collate)
    number_of_node_features = len(train_loader.dataset[0][0].ndata['n_feat'][0])
    number_of_edge_features = len(train_loader.dataset[0][0].edata['e_feat'][0])
    """
    Display the proportion of null iterations (iterations that do not change the cost value of the CVRP solution.
    """
    if 'beta' in keywords_args:
        beta = keywords_args['beta']
    else:
        beta = BETA
    train_weights, training_set_size, test_set_size = compute_classes_weights(train_loader, test_loader, device, beta)

    """
    Create the gated graph convolutional network
    """
    print("#" * 50)
    print("# Date : {0:%y}-{0:%m}-{0:%d}_{0:%H}-{0:%M}".format(datetime.datetime.now()))
    print("# Using {}".format(NETWORK_GATEDGCN))
    if network == NETWORK_GATEDGCN:
        net_params = {
            'in_dim': number_of_node_features,
            'in_dim_edge': number_of_edge_features,
            'hidden_dim': hidden_node_dimensions[0],
            'out_dim': hidden_node_dimensions[-1],
            'n_classes': output_size,
            'dropout': dropout_probability,
            'L': len(hidden_node_dimensions),
            'L_linear': len(hidden_linear_dimensions),
            'readout': 'mean',
            'graph_norm': False,
            'batch_norm': False,
            'residual': False,
            'edge_feat': True,
            'device': device
        }
        graph_convolutional_network = GatedGCNNet(net_params)

        print("# Hidden dimensions : {}".format(net_params['hidden_dim']))
        print("# Number of convolutions layers : {}".format(net_params['L']))
        print("# Number of linear layers : {}".format(net_params['L_linear']))
    else:
        graph_convolutional_network = GCNNet(input_node_features=number_of_node_features,
                                             hidden_node_dimension_list=hidden_node_dimensions,
                                             input_edge_features=number_of_edge_features,
                                             hidden_edge_dimension_list=hidden_edge_dimensions,
                                             hidden_linear_dimension_list=hidden_linear_dimensions,
                                             output_feature=output_size,
                                             dropout_probability=dropout_probability,
                                             device=device)

        print("# Hidden node dimensions : {}".format(hidden_node_dimensions))
        print("# Hidden edge dimensions : {}".format(hidden_edge_dimensions))
        print("# Hidden linear dimensions : {}".format(hidden_linear_dimensions))
    graph_convolutional_network = graph_convolutional_network.to(device)

    """
    Define the optimizer, the learning rate scheduler and the loss function.
    We use the Adam optimizer and a MSE loss.
    """
    optimizer = torch.optim.Adam(graph_convolutional_network.parameters(), lr=initial_learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           patience=patience,
                                                           factor=learning_rate_decrease_factor,
                                                           min_lr=min_learning_rate,
                                                           verbose=True)
    if weight_loss:
        loss_function_training = nn.CrossEntropyLoss(weight=train_weights)
    else:
        loss_function_training = nn.CrossEntropyLoss()
    loss_function_testing = nn.CrossEntropyLoss()
    softmax_function = nn.LogSoftmax(dim=1)
    train_step = make_training_step(graph_convolutional_network,
                                    loss_function_training, softmax_function, optimizer)

    print("# Dropout probability : {}".format(dropout_probability))
    print("# Max epoch : {}".format(max_epoch))
    print("# Initial learning rate : {}".format(initial_learning_rate))
    print("# Device : {}".format(device))
    print("# Training batch size : {}".format(batch_size))
    print("# Testing batch size : {}".format(test_batch_size))
    print("#" * 50)

    """
    Resume training state
    """
    initial_epoch = 0
    training_loss = []
    test_loss = []
    if load_parameters_from_file is not None:
        try:
            training_state = torch.load(MODEL_PARAMETERS_PATH + load_parameters_from_file,
                                        map_location=torch.device(device))
            graph_convolutional_network.load_state_dict(training_state['graph_convolutional_network_state'])
            graph_convolutional_network.train()
            optimizer.load_state_dict(training_state['optimizer_state'])
            initial_epoch = training_state['epoch']
            training_loss = training_state['training_loss']
            test_loss = training_state['test_loss']
            print("Loaded parameters values from {}".format(MODEL_PARAMETERS_PATH + load_parameters_from_file))
            print("Resuming at epoch {}".format(initial_epoch))
        except (pickle.UnpicklingError, TypeError, RuntimeError, KeyError) as exception_value:
            print("Unable to load parameters from {}".format(MODEL_PARAMETERS_PATH + load_parameters_from_file))
            print("Exception : {}".format(exception_value))
            should_continue = ''
            while should_continue != 'y' and should_continue != 'n':
                should_continue = input("Continue anyway with random parameters ? (y/n) ")
            if should_continue == 'n':
                exit(1)

    print("\nStarting training {}\n".format(chr(8987)))

    """
    Train the network.
    """
    for epoch in range(initial_epoch, max_epoch + 1):
        try:
            running_loss = 0.0
            if epoch % DISPLAY_EVERY_N_EPOCH == 1:
                accuracy, test_loss_element, confusion_matrix = evaluate(graph_convolutional_network,
                                                                         loss_function_testing,
                                                                         softmax_function,
                                                                         test_loader,
                                                                         test_set_size)
                test_loss.append(test_loss_element)
                random_accuracy = evaluate_random(test_loader, test_set_size)
                print("Epoch {:d}, loss {:.6f}, test_loss {:.6f}, accuracy {:.4f}, random accuracy {:.4f}"
                      .format(epoch, training_loss[-1], test_loss[-1], accuracy, random_accuracy))
                display_confusion_matrix(confusion_matrix)

            for graph_batch, label_batch in train_loader:
                loss = train_step(graph_batch, label_batch)
                running_loss += loss

            epoch_loss = running_loss / len(train_loader)
            scheduler.step(epoch_loss)
            training_loss.append(epoch_loss)

        except KeyboardInterrupt:
            print("Received keyboard interrupt.")
            print("Computing confusion matrix on training data.")
            _, _, train_confusion_matrix = evaluate(graph_convolutional_network,
                                                    loss_function_testing,
                                                    softmax_function,
                                                    train_loader,
                                                    training_set_size)
            display_confusion_matrix(train_confusion_matrix)
            if save_parameters_on_exit:
                print("Saving parameters before quiting ...", flush=True)
                save_model_parameters(graph_convolutional_network,
                                      optimizer,
                                      str(softmax_function.__class__()).partition('(')[0],
                                      initial_learning_rate, epoch, training_loss, test_loss, device)
            exit(0)

    print("Computing confusion matrix on training data.")
    _, _, train_confusion_matrix = evaluate(graph_convolutional_network,
                                            loss_function_testing,
                                            softmax_function,
                                            train_loader,
                                            training_set_size)
    display_confusion_matrix(train_confusion_matrix)

    if save_parameters_on_exit:
        save_model_parameters(graph_convolutional_network,
                              optimizer,
                              str(softmax_function.__class__()).partition('(')[0],
                              initial_learning_rate, max_epoch, training_loss, test_loss, device)


if __name__ == '__main__':
    recreate = 0
    if recreate:
        main(recreate_dataset=True,
             alns_statistics_file='stats_30it50in.pickle',
             pickle_dataset=True,
             save_parameters_on_exit=False,
             network=NETWORK_GATEDGCN,
             max_epoch=20)
    else:
        main(inputs_labels_name='inputs_labels_'
                                'stats_30it'
                                '50in.pickle',
             network=NETWORK_GATEDGCN,
             save_parameters_on_exit=False,
             max_epoch=25)
