"""Runs PLA Learning Algorithm"""

import sys

from absl import app
from absl import flags
from absl import logging
from matplotlib import pyplot as plt
from generator.bernoulli import LinearBernoulliGenerator, RandomBernoulliGenerator
from algorithm import PLA

FLAGS = flags.FLAGS


def main(argv):
    del argv  # Unused

    logging.info('Running under Python {0[0]}.{0[1]}.{0[2]}'.format(sys.version_info))
    generator = LinearBernoulliGenerator()
    data = generator.generate(1000)

    runner = PLA(data)
    logging.info('Accuracy: {0}%'.format(runner.calculate_accuracy() * 100))

    iteration = 0
    while iteration < 1000:
        runner.iterate()

        iteration += 1
        if iteration % 100 == 0:
            logging.info('Accuracy: {0}%'.format(runner.calculate_accuracy() * 100))
            #draw(runner.data)

    logging.info('DONE Accuracy: {0}%'.format(runner.calculate_accuracy() * 100))
    # plt.show()


def draw(data):
    x_values = []
    y_values = []
    color_values = []

    for point in data:
        x, y = point.features
        target = point.target

        x_values.append(x)
        y_values.append(y)
        color_values.append('red' if target > 0 else 'blue')

    plt.scatter(x_values, y_values, c=color_values)
    plt.show(block=False)


if __name__ == '__main__':
    app.run(main)
