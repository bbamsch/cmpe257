"""Runs PLA Learning Algorithm"""

import numpy as np
import matplotlib.pyplot as plt

from absl import app
from absl import flags
from absl import logging
from generator.bernoulli import LinearBernoulliGenerator, RandomBernoulliGenerator
from algorithm import PLA, Pocket

FLAGS = flags.FLAGS

flags.DEFINE_enum(
    name='generator',
    default='linear',
    enum_values=['linear', 'random'],
    help='Data generator to use')
flags.DEFINE_integer(
    name='learning_points',
    default=1000,
    help='Number of Data points used during learning')
flags.DEFINE_integer(
    name='num_iter',
    default=1000,
    help='Number of PLA iterations to run during training')
flags.DEFINE_integer(
    name='inference_points',
    default=1000,
    help='Number of Data points to classify during inference')
flags.DEFINE_enum(
    name='algorithm',
    default='pla',
    enum_values=['pla', 'pocket'],
    help='Algorithm to use for learning')


def main(argv):
    del argv  # Unused

    logging.info('====== LEARNING ======')
    logging.info('Data Generator => {0}'.format(FLAGS.generator))
    logging.info('Num Data Points => {0}'.format(FLAGS.learning_points))
    logging.info('Num Iterations => {0}'.format(FLAGS.num_iter))

    if FLAGS.generator == 'linear':
        generator = LinearBernoulliGenerator()
    elif FLAGS.generator == 'random':
        generator = RandomBernoulliGenerator()
    else:
        raise NotImplementedError('Generator not found')

    data = generator.generate(FLAGS.learning_points)

    if FLAGS.algorithm == 'pla':
        runner = PLA(data)
    elif FLAGS.algorithm == 'pocket':
        runner = Pocket(data)
    else:
        raise NotImplementedError('Algorithm not found')

    running_accuracy = [runner.calculate_accuracy()]
    for iteration in map(lambda x: x+1, range(FLAGS.num_iter)):
        runner.iterate()

        accuracy = runner.calculate_accuracy()
        running_accuracy.append(accuracy)

        if iteration % 50 == 0:
            logging.info(
                'Iter {0}: Accuracy => {1:.2f}%'.format(
                    iteration,
                    accuracy * 100))

    logging.info('Final: weights => {0}'.format(runner.weights))

    logging.info('====== INFERENCE ======')
    logging.info('Num Data Points => {0}'.format(FLAGS.inference_points))

    test_points = generator.generate(FLAGS.inference_points)
    test_set_total = 0
    test_set_correct = 0
    for point in test_points:
        predicted_target = runner.classify(point)

        if predicted_target == point.target:
            test_set_correct += 1

        test_set_total += 1

    test_set_accuracy = test_set_correct / test_set_total
    logging.info(
        'Model Accuracy on Test Set => {0:.2f}%'.format(
            test_set_accuracy * 100))

    plt.figure(1)
    plot_generator_if_possible(generator)
    plot_model(runner)
    plot_data(data)
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.legend(loc='lower center')
    plt.title(
        'Perceptron (algorithm={0}, generator={1})'.format(
            FLAGS.algorithm,
            FLAGS.generator))
    plt.xlabel('X')
    plt.ylabel('Y')

    plt.figure(2)
    plt.plot(running_accuracy, label='Accuracy')
    plt.xlim(0, FLAGS.learning_points)
    plt.legend(loc='lower center')
    plt.title('Perceptron Accuracy (by iteration)')
    plt.ylabel('Accuracy')
    plt.xlabel('Iteration')
    plt.show()


def plot_generator_if_possible(generator):
    if isinstance(generator, LinearBernoulliGenerator):
        # We can also plot equation of the target line
        m = generator.m
        b = generator.b

        logging.info('Generator weights => {0}'.format([m, b]))

        plt.plot(
            [-1, 1],
            list(map(
                lambda x: (m * x) + b,
                [-1, 1]
            )),
            c='green',
            label='Target')


def plot_model(runner):
    # Extract y = mx + b from model weights
    # Model formula = W0 + W1x + W2y = 0
    # Solve for y:
    # y = -(W1/W2)x - (W0/W2)

    weights = runner.weights
    m = -weights[1]/weights[2]  # m = -(W1/W2)
    b = -weights[0]/weights[2]  # b = -(W0/W2)

    logging.info('Runner weights => {0}'.format(runner.weights))
    logging.info('Model m => {0}'.format(m))
    logging.info('Model b => {0}'.format(b))

    plt.plot(
        [-1, 1],
        list(map(
            lambda x: (m * x) + b,
            [-1, 1]
        )),
        c='orange',
        label='Model')


def plot_data(data: list):
    x_values = []
    y_values = []
    c_values = []

    for point in data:
        x, y = point.features
        target = point.target

        x_values.append(x)
        y_values.append(y)
        c_values.append('red' if target > 0 else 'blue')

    plt.scatter(
        x_values,
        y_values,
        c=c_values)


if __name__ == '__main__':
    app.run(main)
