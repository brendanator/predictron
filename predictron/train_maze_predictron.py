import tensorflow as tf
import time
from datetime import datetime
from . import maze, predictron


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('maze_size', 20, 'Maze size')
tf.app.flags.DEFINE_float('maze_density', 0.3, 'Maze density')
tf.app.flags.DEFINE_string('train_dir', '/tmp/maze_train',
                           'Directory to write event logs and checkpoints')
tf.app.flags.DEFINE_integer('max_steps', 1e6, 'Maximum training steps')
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            'Whether to log device placement.')


def train():
  FLAGS.input_height = FLAGS.maze_size
  FLAGS.input_width = FLAGS.maze_size
  FLAGS.input_channels = 1
  FLAGS.reward_size = FLAGS.maze_size

  with tf.Graph().as_default():
    global_step = tf.contrib.framework.get_or_create_global_step()

    mazes = tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.maze_size,
                                        FLAGS.maze_size, 1])
    labels = tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.maze_size])
    maze_generator = maze.MazeGenerator(height=FLAGS.maze_size,
                                        width=FLAGS.maze_size,
                                        density=FLAGS.maze_density)

    preturns, lambda_preturns = predictron.predictron(mazes)

    preturns_loss, lambda_preturns_loss, consistency_loss = \
                            predictron.loss(preturns, lambda_preturns, labels)
    loss = preturns_loss + lambda_preturns_loss + consistency_loss

    train_op = predictron.train(loss, global_step)

    with tf.train.MonitoredTrainingSession(
        checkpoint_dir=FLAGS.train_dir,
        hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
               tf.train.NanTensorHook(loss),
               LoggerHook(loss)],
        config=tf.ConfigProto(
            log_device_placement=FLAGS.log_device_placement)) as mon_sess:

      while not mon_sess.should_stop():
        _mazes, _labels = maze_generator.generate_batch(FLAGS.batch_size)
        mon_sess.run(train_op, {mazes: _mazes, labels: _labels})


class LoggerHook(tf.train.SessionRunHook):
  """Logs loss and runtime."""

  def __init__(self, loss):
    self._loss = loss

  def begin(self):
    self._step = -1

  def before_run(self, run_context):
    self._step += 1
    self._start_time = time.time()
    return tf.train.SessionRunArgs(self._loss)  # Asks for loss value.

  def after_run(self, run_context, run_values):
    duration = time.time() - self._start_time
    loss_value = run_values.results
    if self._step % 10 == 0:
      num_examples_per_step = FLAGS.batch_size
      examples_per_sec = num_examples_per_step / duration
      sec_per_batch = float(duration)

      format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                    'sec/batch)')
      print (format_str % (datetime.now(), self._step, loss_value,
                            examples_per_sec, sec_per_batch))


def main(_):
  train()


if __name__ == '__main__':
  tf.app.run()
