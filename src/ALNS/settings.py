import os

from project_root_path import get_project_root_path

SEED = 456322

SIZE = 50
CAPACITY = 40
NUMBER_OF_DEPOTS = 1

NUMBER_OF_INSTANCES = 1

ITERATIONS = 1000
WEIGHTS = [1, 1, 1, 1]
OPERATOR_DECAY = 0.8
COLLECT_STATISTICS = True

DEGREE_OF_DESTRUCTION = 0.35
DETERMINISM = 18

START_TEMPERATURE_CONTROL = 0.05
COOLING_RATE = 0.99995
END_TEMPERATURE = 0.01

ROOT_PATH = get_project_root_path()
FILE_PATH = ROOT_PATH \
            + "/data/"\
            + "stats_{0}iter.pickle".format(ITERATIONS)
FILE_MODE = 'wb'
