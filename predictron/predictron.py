import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('predictron_depth', 8, 'Number of layers in predictron')
tf.app.flags.DEFINE_integer('batch_size', 16, 'Batch size')
tf.app.flags.DEFINE_integer('input_height', None, 'Height of input')
tf.app.flags.DEFINE_integer('input_width', None, 'Width of input')
tf.app.flags.DEFINE_integer('input_channels', None, 'Number of input channels')
tf.app.flags.DEFINE_integer('state_kernels', 20, 'Number of state representation kernels')
tf.app.flags.DEFINE_integer('reward_size', None, 'Size of reward vector')
tf.app.flags.DEFINE_boolean('shared_core', True, 'Use a shared core in predictron')


def state_representation(inputs):
  with tf.variable_scope('sr_conv1') as scope:
    kernel = tf.get_variable('weights',
                             [FLAGS.input_height, FLAGS.input_width,
                              FLAGS.input_channels, FLAGS.state_kernels])
    conv = tf.nn.conv2d(inputs, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.get_variable('biases', [FLAGS.state_kernels])
    bias = tf.nn.bias_add(conv, biases)
    conv1 = tf.nn.relu(biases, scope=scope.name)

  with tf.variable_scope('sr_conv2') as scope:
    kernel = tf.get_variable('weights',
                             [FLAGS.input_height, FLAGS.input_width,
                              FLAGS.state_kernels, FLAGS.state_kernels])
    conv = tf.nn.conv2d(conv1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.get_variable('biases', [FLAGS.state_kernels])
    bias = tf.nn.bias_add(conv, biases)
    conv2 = tf.nn.relu(biases, scope=scope.name)

  return conv2


def model_network(state):
  with tf.variable_scope('model') as scope:
    kernel = tf.get_variable('weights',
                             [FLAGS.input_height, FLAGS.input_width,
                              FLAGS.state_kernels, FLAGS.state_kernels])
    conv = tf.nn.conv2d(state, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.get_variable('biases', [FLAGS.state_kernels])
    bias = tf.nn.bias_add(conv, biases)
    next_state = tf.nn.relu(biases, scope=scope.name)


def predict_states(state):
  states = [state]

  with tf.variable_scope('states'):
    for i in range(1, FLAGS.predictron_depth+1):
      with tf.variable_scope('layer-%d' % i):
        state = model_network(state)
        states.append(state)

    return state.pack(states)


def state_size():
  return FLAGS.input_height * FLAGS.input_width * FLAGS.state_kernels


def states_to_output_network(states, scope):
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

  logits = tf.reshape(logits,
                      [FLAGS.batch_size, FLAGS.predictron_depth, FLAGS.reward_size])
  return tf.nn.relu(logits, name=scope.name)


def reward_network(states):
  with tf.variable_scope('reward') as scope:
    rewards = states_to_output_network(states, scope)

    # Set rewards[0] to zero
    rewards = tf.slice(rewards, [0, 0, 0], [FLAGS.batch_size, FLAGS.predictron_depth-1, FLAGS.reward_size])
    rewards = tf.concat(1, [tf.zeros([FLAGS.batch_size, 1, FLAGS.reward_size]),
                            rewards])
    return rewards


def discount_network(states):
  with tf.variable_scope('discount') as scope:
    discounts = states_to_output_network(states, scope)

    # Set discounts[0] to one
    discounts = tf.slice(discounts, [0, 0, 0], [FLAGS.batch_size, FLAGS.predictron_depth-1, FLAGS.reward_size])
    discounts = tf.concat(1, [tf.ones([FLAGS.batch_size, 1, FLAGS.reward_size]),
                              discounts])
    return discounts


def value_network(states):
  with tf.variable_scope('value') as scope:
    return states_to_output_network(states, scope)


def lambda_network(states):
  with tf.variable_scope('lambda') as scope:
    lambdas = states_to_output_network(states, scope)

    # Set lambdas[-1] to one
    lambdas = tf.slice(lambdas, [0, 0, 0], [FLAGS.batch_size, FLAGS.predictron_depth-1, FLAGS.reward_size])
    lambdas = tf.concat(1, [lambdas,
                            tf.ones([FLAGS.batch_size, 1, FLAGS.reward_size])])
    return lambdas



# def predictron(state):
#   # Accumulate rewards, discounts and lambdas and each prediction step
#   rewards, lambdas = [], []
#   # First discount must always be 1
#   discounts = [tf.ones([FLAGS.batch_size, state_size()], dtype=tf.float32)]

#   # Offsets are not quite right
#   # We;re passing in state_0

#   # Also need to return values

#   # Should every layer really have different reward/discount/lambda networks or is it just state?

#   # Is the best way to do this
#   #  1. Just returns all the predicted states from here
#   #  2. Accumulate outputs in separate functions?
#   # This really depends on the above question, or does it

#   with tf.variable_scope('predictron'):
#     for i in range(FLAGS.predictron_depth):
#       with tf.variable_scope('layer-%d' % i):
#         reward = reward_network(state)
#         discount = discount_network(state)
#         # Final lambda must always be 0
#         if i < FLAGS.predictron_depth:
#           lambda_ = lambda_network(state)
#         else:
#           lambda_ = tf.zeros([FLAGS.batch_size, state_size()], dtype=tf.float32)
#         state = model_network(state)

#         rewards.append(reward)
#         discounts.append(discount)
#         lambdas.append(lambda_)


#     rewards = tf.pack(rewards)
#     discounts = tf.pack(discounts)
#     lambdas = tf.pack(lambdas)

#   return rewards, discounts, lambdas


def preturn(values, rewards, discounts):
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

  return preturns


def lambda_predictron(preturns, lambdas):
  # Final lamdba must be 0
  final_lambda = tf.Assert(tf.reduce_all(tf.equal(lambdas[:, -1, :], 0.0)), [lambdas[:, -1, :]])

  with tf.control_dependencies([final_lambda]):
    with tf.variable_scope('lambda_predictron'):
      accum_lambda = tf.cumprod(lambdas, axis=1, exclusive=True)
      lambda_bar = (1 - lambdas) * accum_lambda # This should always sum to 1
      lambda_preturn = tf.reduce_sum(lambda_bar * preturns)

  return lambda_preturn
