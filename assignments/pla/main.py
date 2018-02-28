"""Runs PLA Learning Algorithm"""

import sys

from absl import app
from absl import flags
from absl import logging
from matplotlib import pyplot as plt
from generator.bernoulli.linear import LinearBernoulliGenerator


FLAGS = flags.FLAGS


def main(argv):
    del argv  # Unused

    logging.info('Running under Python {0[0]}.{0[1]}.{0[2]}'.format(sys.version_info))
    generator = LinearBernoulliGenerator()
    data = generator.generate(1000)
    logging.info(data)

    x_values = []
    y_values = []
    color_values = []
    for point in data:
        x, y = point[0]
        target = point[1]

        x_values.append(x)
        y_values.append(y)
        color_values.append('red' if target else 'blue')

    plt.scatter(x_values, y_values, c=color_values)

    m: float = generator.get_m()
    b: float = generator.get_b()
    plt.plot([-1, 1], [m * -1 + b, m * 1 + b])
    plt.show()


if __name__ == '__main__':
    app.run(main)
