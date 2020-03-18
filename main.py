import os
import time

from ALNS.generate_alns_stats import generate_stats
from NeuralNetwork.create_dataset import create_dataset

import ALNS.settings as settings


def main():
    start_time = time.clock()
    generate_stats(settings.FILE_NAME, 1)
    x, y = create_dataset(settings.FILE_NAME)
    duration = time.clock() - start_time
    print("Duration: {4}\n"
          "Determinism : {0}\n"
          "Number of non-zero deltas : {1}\n"
          "Length of stats : {2} (equal to {3})"
          .format(settings.DETERMINISM, len(y) - y.count(0.0), len(y), settings.ITERATIONS, duration))

    os.remove(settings.FILE_NAME)
    os.system('aplay -d 2 /usr/share/sounds/ringtone-internet1.wav')


if __name__ == '__main__':
    main()
