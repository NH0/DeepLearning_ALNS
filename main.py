import os
import time

import ALNS.settings as settings

from ALNS.generate_alns_stats import generate_stats
from NeuralNetwork.create_dataset import create_dataset


def main():
    start_time = time.clock()
    generate_stats(settings.FILE_PATH, 1)
    x, y = create_dataset(settings.FILE_PATH)
    duration = time.clock() - start_time
    number_of_non_zero_deltas = settings.ITERATIONS - y.count(0.0)
    print("Duration: {3:.2f} seconds\n"
          "Determinism : {0}\n"
          "Number of non-zero deltas : {1} ({4:.2f}% of total)\n"
          "Length of stats : {2}"
          .format(settings.DETERMINISM, number_of_non_zero_deltas, settings.ITERATIONS, duration,
                  100 * number_of_non_zero_deltas / settings.ITERATIONS))

    # os.remove(settings.FILE_NAME)
    # os.system('aplay -d 2 /usr/share/sounds/ringtone-internet1.wav')


if __name__ == '__main__':
    main()
