import tensorflow as tf
import numpy as np

from datetime import datetime
import time

from . import maze, predictron

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('maze_size', 20, 'Maze size')
tf.app.flags.DEFINE_float('maze_density', 0.3, 'Maze density')
tf.app.flags.DEFINE_string('train_dir', '/tmp/maze_train',
                           'Directory to write event logs and checkpoints')
tf.app.flags.DEFINE_integer('max_steps', 1e6, 'Maximum training steps')
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            'Whether to log device placement.')
tf.app.flags.DEFINE_integer(
    'consistency_updates', 0,
    'Number of semi supervised constistency updates to perform')


def train():
  FLAGS.input_height = FLAGS.maze_size
  FLAGS.input_width = FLAGS.maze_size
  FLAGS.input_channels = 1
  FLAGS.reward_size = FLAGS.maze_size
  FLAGS.is_training = True

  with tf.Graph().as_default():
    global_step = tf.contrib.framework.get_or_create_global_step()

    mazes = tf.placeholder(
        tf.float32, [FLAGS.batch_size, FLAGS.maze_size, FLAGS.maze_size, 1])
    labels = tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.maze_size])

    generator = maze.MazeGenerator(
        height=FLAGS.maze_size,
        width=FLAGS.maze_size,
        density=FLAGS.maze_density)

    preturns, lambda_preturn = predictron.predictron(mazes, FLAGS)

    preturns_loss, lambda_preturn_loss, consistency_loss = \
                            predictron.loss(preturns, lambda_preturn, labels)
    total_loss = preturns_loss + lambda_preturn_loss + consistency_loss

    train_op, semi_supervised_train = predictron.train(
        total_loss, consistency_loss, global_step)

    with tf.train.MonitoredTrainingSession(
        checkpoint_dir=FLAGS.train_dir,
        hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps)],
        config=tf.ConfigProto(
            log_device_placement=FLAGS.log_device_placement)) as mon_sess:

      step = 0
      while not mon_sess.should_stop():
        step += 1
        start_time = time.time()

        # Supervised learning
        _mazes, _labels = generator.generate_labelled_batch(FLAGS.batch_size)
        feed_dict = {mazes: _mazes, labels: _labels}
        loss_value, _ = mon_sess.run([total_loss, train_op], feed_dict)
        check_nan(loss_value)

        # Semi-supervised learning
        for _ in range(FLAGS.consistency_updates):
          _mazes = generator.generate_batch(FLAGS.batch_size)
          consistency_loss_value, _ = mon_sess.run(
              [consistency_loss, semi_supervised_train], {mazes: _mazes})
          check_nan(consistency_loss_value)

        log_step(step, start_time, loss_value)


def check_nan(loss_value):
  if np.isnan(loss_value):
    tf.logging.error('Model diverged with loss = NaN')
    raise tf.train.NanLossDuringTrainingError


def log_step(step, start_time, loss_value):
  if step % 10 == 0:
    duration = time.time() - start_time
    examples_per_sec = FLAGS.batch_size * (
        FLAGS.consistency_updates + 1) / duration
    sec_per_batch = float(duration)
    format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                  'sec/step)')
    print(format_str % (datetime.now(), step, loss_value, examples_per_sec,
                        sec_per_batch))


def main(_):
  train()


if __name__ == '__main__':
  tf.app.run()
