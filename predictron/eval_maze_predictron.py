from __future__ import print_function

import tensorflow as tf
import numpy as np

from datetime import datetime
import math
import time

from . import maze, predictron, train_maze_predictron
from .util import Colour

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('eval_dir', '/tmp/maze_eval',
                           'Directory where to write event logs.')
tf.app.flags.DEFINE_string('checkpoint_dir', '/tmp/maze_train',
                           'Directory where to read model checkpoints.')
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 1,
                            'How often to run the eval.')
tf.app.flags.DEFINE_integer('num_examples', 10000,
                            'Number of examples to run.')
tf.app.flags.DEFINE_boolean('run_once', False,
                            'Whether to run eval only once.')


def eval_once(saver, mazes, labels, generator, diagonal_predictions,
              correct_predictions):
  with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      # Restores from checkpoint
      saver.restore(sess, ckpt.model_checkpoint_path)
      # Assuming model_checkpoint_path looks something like:
      #   /my-favorite-path/cifar10_train/model.ckpt-0,
      # extract global_step from it.
      global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    else:
      print('No checkpoint file found')
      return

    num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
    true_count = 0  # Counts the number of correct predictions.
    total_sample_count = num_iter * FLAGS.batch_size
    step = 0
    while step < num_iter:
      _mazes, _labels = generator.generate_labelled_batch(FLAGS.batch_size)
      feed_dict = {mazes: _mazes, labels: _labels}
      predictions, correct = sess.run(
          [diagonal_predictions, correct_predictions], feed_dict)
      true_count += np.sum(correct)
      step += 1

    # Compute precision
    precision = true_count / total_sample_count
    print('%s: Precision = %.3f' % (datetime.now(), precision))

    # Print sample maze and predictions
    print('Sample maze:')
    generator.print_maze(_mazes[0], _labels[0])
    print('Predictions:')
    predictions = [int(prediction) for prediction in predictions[0]]
    for prediction, label in zip(predictions, _labels[0]):
      print(Colour.highlight(str(prediction), prediction == label), end='')
    print()


def evaluate():
  FLAGS.input_height = FLAGS.maze_size
  FLAGS.input_width = FLAGS.maze_size
  FLAGS.input_channels = 1
  FLAGS.reward_size = FLAGS.maze_size
  FLAGS.is_training = False

  with tf.Graph().as_default() as g:
    mazes = tf.placeholder(
        tf.float32, [FLAGS.batch_size, FLAGS.maze_size, FLAGS.maze_size, 1])
    labels = tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.maze_size])
    generator = maze.MazeGenerator(
        height=FLAGS.maze_size,
        width=FLAGS.maze_size,
        density=FLAGS.maze_density)
    preturns, lambda_preturn = predictron.predictron(mazes, FLAGS)

    diagonal_predictions = tf.round(lambda_preturn)
    correct_predictions = tf.reduce_mean(
        tf.to_float(tf.equal(diagonal_predictions, labels)),
        reduction_indices=1)

    # Restore the moving average version of the learned variables for eval.
    variable_averages = tf.train.ExponentialMovingAverage(0.9999)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.summary.merge_all()

    summary_writer = tf.summary.FileWriter(FLAGS.eval_dir, g)

    while True:
      eval_once(saver, mazes, labels, generator, diagonal_predictions,
                correct_predictions)
      if FLAGS.run_once:
        break
      time.sleep(FLAGS.eval_interval_secs)


def main(_):
  evaluate()


if __name__ == '__main__':
  tf.app.run()
