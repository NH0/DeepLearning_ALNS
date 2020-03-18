import os

from ALNS.generate_alns_stats import generate_stats
from NeuralNetwork.create_dataset import create_dataset

import ALNS.settings as settings

generate_stats(settings.FILE_NAME, 1)
x, y = create_dataset(settings.FILE_NAME)
print("Determinism : {0}\n"
      "Number of zero deltas : {1}\n"
      "Lenght of stats : {2} (equal to {3})"\
      .format(settings.DETERMINISM, y.count(0.0), len(y), settings.ITERATIONS))

os.remove(settings.FILE_NAME)
