"""Runs PLA Learning Algorithm"""

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

    for iteration in map(lambda x: x+1, range(FLAGS.num_iter)):
        runner.iterate()

        if iteration % 50 == 0:
            logging.info(
                'Iter {0}: Accuracy => {1:.2f}%'.format(
                    iteration,
                    runner.calculate_accuracy() * 100))

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


if __name__ == '__main__':
    app.run(main)
