from project_root_path import get_project_root_path
from torch.cuda import is_available as is_cuda_available

ROOT_PATH = get_project_root_path()
STATISTICS_DATA_PATH = ROOT_PATH + '/data/'
INPUTS_LABELS_PATH = STATISTICS_DATA_PATH
MODEL_PARAMETERS_PATH = STATISTICS_DATA_PATH

ALNS_STATISTICS_FILE = 'stats_20it2000in.pickle'
INPUTS_LABELS_PREFIX = 'inputs_labels_'
# for search&replace, here is the dataset name in the form of a string :
# old : inputs_mask_labels_dataset_50-50_1inst_50nod_40cap_1dep_50000iter_0.8decay_0.35destr_18determ.pickle
# old : dataset_50-50_stats_50000iter.pickle
# old : inputs_labels_50-50_stats_50000iter.pickle
# inputs_labels_stats_20it2000in.pickle
INPUTS_LABELS_NAME = INPUTS_LABELS_PREFIX + ALNS_STATISTICS_FILE

NETWORK_GCN = 'GCNNet'
NETWORK_GATEDGCN = 'GatedGCNNet'
NETWORK_PARAMS_FILE = 'GCNparams_ep91_lr0.001_devcuda_LogSoftmax.pt'

HIDDEN_NODE_DIMENSIONS = [64, 64, 64, 64]
HIDDEN_EDGE_DIMENSIONS = [64, 64, 64, 64]
HIDDEN_LINEAR_DIMENSIONS = [32, 16, 8]
OUTPUT_SIZE = 3
DROPOUT_PROBABILITY = 0.0
MAX_EPOCH = 5000
EPSILON = 1e-8

INITIAL_LEARNING_RATE = 0.001
MIN_LEARNING_RATE = 5 * 1e-6
LEARNING_RATE_DECREASE_FACTOR = 0.6
PATIENCE = 100

BETA = 0.99

MASK_SEED = 123456
BATCH_SIZE = 64

DEVICE = 'cuda' if is_cuda_available() else 'cpu'

DISPLAY_EVERY_N_EPOCH = 5

if __name__ == '__main__':
    print(ROOT_PATH)
