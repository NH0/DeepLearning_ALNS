import re
import sys

import matplotlib.pyplot as plt

from numpy import arange
from project_root_path import get_project_root_path


def parse_result(result_file_path):
    loss = []
    test_loss = []

    hidden_node_dimensions = ""
    hidden_edge_dimensions = ""
    hidden_linear_dimension = ""

    display_every_n_epoch = -1
    has_display_every_n_epoch = False
    start_epoch = -1

    with open(result_file_path, 'r') as result_file:
        for line in result_file.readlines():
            if line.startswith("# Hidden node dimensions : "):
                hidden_node_dimensions = re.search(r"\[(\d+, )*\d+\]$", line).group()
            elif line.startswith("# Hidden edge dimensions : "):
                hidden_edge_dimensions = re.search(r"\[(\d+, )*\d+\]$", line).group()
            elif line.startswith("# Hidden linear dimension"):
                hidden_linear_dimension = re.search(r"(\[(\d+, )*\d+\]|\d+)$", line).group()
            elif line.startswith("Epoch"):
                if not has_display_every_n_epoch:
                    search = re.search(r"Epoch (\d+),", line)
                    if search is not None:
                        epoch = int(search.group(1))
                    else:
                        continue
                    if start_epoch == -1:
                        start_epoch = epoch
                    else:
                        display_every_n_epoch = epoch - start_epoch
                        has_display_every_n_epoch = True
                loss.append(float(re.search(r"loss (-?[0-9]+\.[0-9]+)", line).group(1)))
                test_loss_epoch = re.search(r"test_loss (-?[0-9]+\.[0-9]+)", line).group(1)
                if test_loss_epoch is not None:
                    test_loss.append(float(test_loss_epoch))

    return {'loss': loss,
            'test_loss': test_loss,
            'hidden_node_dimensions': hidden_node_dimensions,
            'hidden_edge_dimensions': hidden_edge_dimensions,
            'hidden_linear_dimension': hidden_linear_dimension,
            'display_every_n_epoch': display_every_n_epoch,
            'start_epoch': start_epoch}


def display_loss(result_file):
    result_file_path = get_project_root_path() + '/results/' + result_file
    result = parse_result(result_file_path)

    loss = result['loss']
    test_loss = result['test_loss']

    display_every_n_epoch = result['display_every_n_epoch']
    start_epoch = result['start_epoch']
    end_epoch = display_every_n_epoch * len(loss)
    epochs = arange(start_epoch, end_epoch, display_every_n_epoch)

    ax = plt.axes()
    plt.plot(epochs, loss, label='train loss')
    if len(test_loss) > 0:
        plt.plot(epochs, test_loss, label='test loss')
    # x_epsilon = end_epoch * 1/30
    # y_epsilon = max(loss) * 1/30
    # plt.axis([start_epoch - x_epsilon, end_epoch + x_epsilon, min(loss) - y_epsilon, max(loss) + y_epsilon])
    plt.legend(loc='upper left')
    plt.title(result_file)
    plt.ylabel("Loss value (NLL loss)")
    plt.xlabel("Epoch")
    plt.text(0.98, 0.97,
             'Hidden node dimensions : ' + result['hidden_node_dimensions']
             + '\nHidden edge dimensions : ' + result['hidden_edge_dimensions']
             + '\nHidden linear dimensions : ' + result['hidden_linear_dimension'],
             transform=ax.transAxes,
             horizontalalignment='right',
             verticalalignment='top',
             multialignment='left',
             bbox=dict(facecolor='white', alpha=0.7))

    plt.savefig(result_file_path + '.png')
    plt.show()


def main(loss_file):
    display_loss(loss_file)


if __name__ == '__main__':
    main(sys.argv[1])
