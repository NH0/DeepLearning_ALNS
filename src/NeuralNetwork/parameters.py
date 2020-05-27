from project_root_path import get_project_root_path

ROOT_PATH = get_project_root_path()
DATASET_PREFIX = 'inputs_mask_labels_'
STATISTICS_DATA_PATH = ROOT_PATH + '/data/'
ALNS_STATISTICS_FILE = 'dataset_50-50_1inst_50nod_40cap_1dep_50000iter_0.8decay_0.35destr_18determ.pickle'
DATASET_PATH = STATISTICS_DATA_PATH + DATASET_PREFIX + ALNS_STATISTICS_FILE
MODEL_PARAMETERS_PATH = STATISTICS_DATA_PATH + 'parametersGCN'

HIDDEN_NODE_DIMENSIONS = [512, 256, 256, 128]
HIDDEN_EDGE_DIMENSIONS = [256, 128, 128, 64]
HIDDEN_LINEAR_DIMENSIONS = [128, 128, 32, 8]
OUTPUT_SIZE = 3
DROPOUT_PROBABILITY = 0.2
MAX_EPOCH = 5000
EPSILON = 1e-5

INITIAL_LEARNING_RATE = 0.00001
LEARNING_RATE_DECREASE_FACTOR = 0.9

MASK_SEED = 123456

DISPLAY_EVERY_N_EPOCH = 5

if __name__ == '__main__':
    print(ROOT_PATH)
