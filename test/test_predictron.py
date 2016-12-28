import tensorflow as tf

import predictron.predictron as predictron


class PredictronTest(tf.test.TestCase):
  def test_preturn(self):
    with self.test_session() as sess:
      rewards = tf.constant([0, 1, 2], shape=[1, 3, 1], dtype=tf.float32)
      discounts = tf.constant(
          [1.0, 0.5, 0.2], shape=[1, 3, 1], dtype=tf.float32)
      values = tf.constant([10, 8, 5], shape=[1, 3, 1], dtype=tf.float32)

      preturn = predictron.preturn_network(rewards, discounts, values)

      p = sess.run(preturn)

      self.assertAllClose(p, [[[10], [5], [2.5]]])

  def test_lambda_predictron(self):
    with self.test_session() as sess:
      preturns = tf.constant([1, 2, 3], shape=[1, 3, 1], dtype=tf.float32)
      lambdas = tf.constant([0.5, 0.4, 0.0], shape=[1, 3, 1], dtype=tf.float32)

      lambda_predictron = predictron.lambda_preturn_network(preturns, lambdas)

      lp = sess.run(lambda_predictron)

      self.assertNear(lp, 1.7, 1e-6)

  def test_state_to_output_network(self):
    config = tf.app.flags.FLAGS
    config.batch_size = 16
    config.predictron_depth = 8
    config.input_height = 10
    config.input_width = 5
    config.reward_size = 3

    for shared_core in [True, False]:
      with tf.variable_scope('shared-%s' % str(shared_core)) as scope:
        config.shared_core = shared_core

        states = tf.zeros([
            config.batch_size, config.predictron_depth, config.input_height,
            config.input_width, config.state_kernels
        ])

        rewards = predictron.reward_network(states, config)

        self.assertAllEqual(rewards.get_shape().as_list(), [16, 8, 3])


if __name__ == "__main__":
  tf.test.main()
