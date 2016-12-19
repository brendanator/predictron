import tensorflow as tf
from . import util


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('batch_size', 16, 'Batch size')
tf.app.flags.DEFINE_integer('input_height', None, 'Height of input')
tf.app.flags.DEFINE_integer('input_width', None, 'Width of input')
tf.app.flags.DEFINE_integer('input_channels', None, 'Number of input channels')
tf.app.flags.DEFINE_integer('state_kernels', 20, 'Number of state representation kernels')
tf.app.flags.DEFINE_integer('predictron_depth', 8, 'Number of layers in predictron')
tf.app.flags.DEFINE_boolean('shared_core', True, 'Use a shared core in predictron')
tf.app.flags.DEFINE_integer('reward_size', None, 'Size of reward vector')


def predictron(inputs):
  state = state_representation(inputs)
  predicted_states = predict_states(state)
  rewards = reward_network(predicted_states)
  discounts = discount_network(predicted_states)
  values = value_network(predicted_states)
  lambdas = lambda_network(predicted_states)
  preturns = preturn_network(rewards, discounts, values)
  lambda_preturns = lambda_preturn_network(preturns, lambdas)
  return preturns, lambda_preturns


def loss(preturns, lambda_preturns, labels):
  with tf.variable_scope('loss'):
    preturns_loss = tf.reduce_mean(tf.squared_difference(preturns, tf.expand_dims(labels, dim=1)))
    lambda_preturns_loss = \
              tf.reduce_mean(tf.squared_difference(lambda_preturns, labels))
    consistency_loss = tf.reduce_mean(tf.squared_difference(preturns, tf.stop_gradient(tf.expand_dims(lambda_preturns, dim=1))))
    return preturns_loss, lambda_preturns_loss, consistency_loss


def train(total_loss, global_step):
  # Generate moving averages of all losses and associated summaries.
  loss_averages_op = util.add_loss_summaries(total_loss)

  # Compute gradients.
  with tf.control_dependencies([loss_averages_op]):
    opt = tf.train.AdamOptimizer()
    grads = opt.compute_gradients(total_loss)

  # Apply gradients.
  apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

  # Add histograms for trainable variables.
  for var in tf.trainable_variables():
    tf.summary.histogram('trainable', var)

  # Add histograms for gradients.
  for grad, var in grads:
    if grad is not None:
      tf.summary.histogram('gradients', grad)

  # Track the moving averages of all trainable variables.
  variable_averages = tf.train.ExponentialMovingAverage(
      decay=0.9999, num_updates=global_step)
  variables_averages_op = variable_averages.apply(tf.trainable_variables())

  with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
    train_op = tf.no_op(name='train')

  return train_op



def state_representation(inputs):
  with tf.variable_scope('sr_conv1') as scope:
    kernel = tf.get_variable('weights',
                             [FLAGS.input_height, FLAGS.input_width,
                              FLAGS.input_channels, FLAGS.state_kernels])
    conv = tf.nn.conv2d(inputs, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.get_variable('biases', [FLAGS.state_kernels])
    bias = tf.nn.bias_add(conv, biases)
    conv1 = tf.nn.relu(bias, name=scope.name)
    util.activation_summary(conv1)

  with tf.variable_scope('sr_conv2') as scope:
    kernel = tf.get_variable('weights',
                             [FLAGS.input_height, FLAGS.input_width,
                              FLAGS.state_kernels, FLAGS.state_kernels])
    conv = tf.nn.conv2d(conv1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.get_variable('biases', [FLAGS.state_kernels])
    bias = tf.nn.bias_add(conv, biases)
    conv2 = tf.nn.relu(bias, name=scope.name)
    util.activation_summary(conv2)

  return conv2


def model_network(state):
  with tf.variable_scope('model') as scope:
    kernel = tf.get_variable('weights',
                             [FLAGS.input_height, FLAGS.input_width,
                              FLAGS.state_kernels, FLAGS.state_kernels])
    conv = tf.nn.conv2d(state, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.get_variable('biases', [FLAGS.state_kernels])
    bias = tf.nn.bias_add(conv, biases)
    next_state = tf.nn.relu(bias, name=scope.name)
    util.activation_summary(next_state)

  return next_state


def predict_states(state):
  states = [state]

  with tf.variable_scope('states'):
    for i in range(1, FLAGS.predictron_depth):
      with tf.variable_scope('layer-%d' % i):
        state = model_network(state)
        states.append(state)

    return tf.pack(states, axis=1)


def state_size():
  return FLAGS.input_height * FLAGS.input_width * FLAGS.state_kernels


def states_to_output_network(states):
  if FLAGS.shared_core:
    weights = tf.get_variable('weights', [state_size(), FLAGS.reward_size])
    biases = tf.get_variable('biases', [FLAGS.reward_size])
    states = tf.reshape(states, [FLAGS.batch_size * FLAGS.predictron_depth, state_size()])
    logits = tf.nn.xw_plus_b(states, weights, biases)
  else:
    weights = tf.get_variable('weights',
                              [1, FLAGS.predictron_depth, state_size(), FLAGS.reward_size])
    weights = tf.tile(weights, [FLAGS.batch_size, 1, 1, 1]) # Manual broadcasting
    biases = tf.get_variable('biases', [FLAGS.predictron_depth, FLAGS.reward_size])
    states = tf.reshape(states, [FLAGS.batch_size, FLAGS.predictron_depth, 1, state_size()])
    logits = tf.squeeze(tf.batch_matmul(states, weights)) + biases

  return tf.reshape(logits,
                      [FLAGS.batch_size, FLAGS.predictron_depth, FLAGS.reward_size])


def reward_network(states):
  with tf.variable_scope('reward') as scope:
    rewards = states_to_output_network(states)

    # Set rewards[0] to zero
    rewards = tf.slice(rewards, [0, 0, 0], [FLAGS.batch_size, FLAGS.predictron_depth-1, FLAGS.reward_size])
    rewards = tf.concat(1, [tf.zeros([FLAGS.batch_size, 1, FLAGS.reward_size]),
                            rewards])

    util.activation_summary(rewards)
    return rewards


def discount_network(states):
  with tf.variable_scope('discount') as scope:
    logits = states_to_output_network(states)
    discounts = tf.nn.sigmoid(logits)

    # Set discounts[0] to one
    discounts = tf.slice(discounts, [0, 0, 0], [FLAGS.batch_size, FLAGS.predictron_depth-1, FLAGS.reward_size])
    discounts = tf.concat(1, [tf.ones([FLAGS.batch_size, 1, FLAGS.reward_size]),
                              discounts])

    util.activation_summary(discounts)
    return discounts


def value_network(states):
  with tf.variable_scope('value') as scope:
    values = states_to_output_network(states)
    util.activation_summary(values)
    return values


def lambda_network(states):
  with tf.variable_scope('lambda') as scope:
    logits = states_to_output_network(states)
    lambdas = tf.nn.sigmoid(logits, name=scope.name)

    # Set lambdas[-1] to zero
    lambdas = tf.slice(lambdas, [0, 0, 0], [FLAGS.batch_size, FLAGS.predictron_depth-1, FLAGS.reward_size])
    lambdas = tf.concat(1, [lambdas,
                            tf.zeros([FLAGS.batch_size, 1, FLAGS.reward_size])])

    util.activation_summary(lambdas)
    return lambdas


def preturn_network(rewards, discounts, values):
  # First reward must be zero, first discount must be one
  first_reward = tf.Assert(tf.reduce_all(tf.equal(rewards[:, 0, :], 0.0)), [rewards[:, 0, :]])
  first_discount = tf.Assert(tf.reduce_all(tf.equal(discounts[:, 0, :], 1.0)), [discounts[:, 0, :]])

  with tf.control_dependencies([first_reward, first_discount]):
    with tf.variable_scope('preturn'):
      accum_value_discounts = tf.cumprod(discounts, axis=1, exclusive=False)
      accum_reward_discounts = tf.cumprod(discounts, axis=1, exclusive=True)
      discounted_values = values * accum_value_discounts
      discounted_rewards = rewards * accum_reward_discounts
      cumulative_rewards = tf.cumsum(discounted_rewards, axis=1)
      preturns = cumulative_rewards + discounted_values
      util.activation_summary(preturns)

  return preturns


def lambda_preturn_network(preturns, lambdas):
  # Final lamdba must be zero
  final_lambda = tf.Assert(tf.reduce_all(tf.equal(lambdas[:, -1, :], 0.0)), [lambdas[:, -1, :]])

  with tf.control_dependencies([final_lambda]):
    with tf.variable_scope('lambda_predictron'):
      accum_lambda = tf.cumprod(lambdas, axis=1, exclusive=True)
      lambda_bar = (1 - lambdas) * accum_lambda # This should always sum to 1
      lambda_preturns = tf.reduce_sum(lambda_bar * preturns, reduction_indices=1)
      util.activation_summary(lambda_preturns)

  return lambda_preturns
