import tensorflow as tf
from . import util

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('batch_size', 100, 'Batch size')
tf.app.flags.DEFINE_integer('input_height', None, 'Height of input')
tf.app.flags.DEFINE_integer('input_width', None, 'Width of input')
tf.app.flags.DEFINE_integer('input_channels', None, 'Number of input channels')
tf.app.flags.DEFINE_integer('state_kernels', 32,
                            'Number of state representation kernels')
tf.app.flags.DEFINE_integer('predictron_depth', 8,
                            'Number of layers in predictron')
tf.app.flags.DEFINE_integer('output_hidden_size', 32,
                            'Size of hidden layer in output networks')
tf.app.flags.DEFINE_boolean('shared_core', True,
                            'Use a shared core in predictron')
tf.app.flags.DEFINE_integer('reward_size', None, 'Size of reward vector')


def predictron(inputs, config):
  state = state_representation(inputs, config)
  states, hidden_states = rollout_states(state, config)
  values = value_network(states, config)
  rewards = reward_network(hidden_states, config)
  discounts = discount_network(hidden_states, config)
  lambdas = lambda_network(hidden_states, config)
  preturns = preturn_network(rewards, discounts, values)
  lambda_preturn = lambda_preturn_network(preturns, lambdas)
  return preturns, lambda_preturn


def loss(preturns, lambda_preturn, labels):
  with tf.variable_scope('loss'):
    preturns_loss = tf.reduce_mean(
        tf.squared_difference(preturns, tf.expand_dims(labels, 1)))

    lambda_preturn_loss = tf.reduce_mean(
        tf.squared_difference(lambda_preturn, labels))

    consistency_loss = tf.reduce_mean(
        tf.squared_difference(
            preturns, tf.stop_gradient(tf.expand_dims(lambda_preturn, 1))))

    l2_loss = tf.get_collection('losses')

    total_loss = preturns_loss + lambda_preturn_loss + consistency_loss
    consistency_loss += l2_loss
    return total_loss, consistency_loss


def train(total_loss, consistency_loss, global_step):
  # Generate moving averages of all losses and associated summaries.
  loss_averages_op = util.add_loss_summaries(total_loss)

  # Optimizer
  opt = tf.train.AdamOptimizer()

  # Compute gradients.
  with tf.control_dependencies([loss_averages_op]):
    grads = opt.compute_gradients(total_loss)

  # Apply gradients.
  apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

  # Add histograms for trainable variables.
  for var in tf.trainable_variables():
    tf.summary.histogram('trainable', var)

  # Add histograms for gradients.
  for grad, var in grads:
    if grad is not None:
      tf.summary.histogram('gradient', grad)

  # Track the moving averages of all trainable variables.
  variable_averages = tf.train.ExponentialMovingAverage(
      decay=0.9999, num_updates=global_step)
  variables_averages_op = variable_averages.apply(tf.trainable_variables())

  # Get batch_norm updates to run during training
  update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

  # Training
  with tf.control_dependencies([apply_gradient_op, variables_averages_op] +
                               update_ops):
    train_op = tf.no_op(name='train')

  # Semi supervised training
  semi_supervised_grads = opt.compute_gradients(consistency_loss)
  semi_supervised_apply_gradient_op = opt.apply_gradients(
      semi_supervised_grads)
  for grad, var in semi_supervised_grads:
    if grad is not None:
      tf.summary.histogram('semi_supervised_gradient', grad)

  with tf.control_dependencies([semi_supervised_apply_gradient_op] +
                               update_ops):
    semi_supervised_train = tf.no_op(name='semi_supervised_train')

  return train_op, semi_supervised_train


def state_representation(inputs, config):
  with tf.variable_scope('state_representation') as scope:
    with tf.variable_scope('layer-1') as scope:
      kernel_1 = util.variable_with_weight_decay(
          'weights', [3, 3, config.input_channels, config.state_kernels])
      biases_1 = util.variable_on_cpu('biases', [config.state_kernels],
                                      tf.constant_initializer(0.1))
      conv_1 = tf.nn.conv2d(inputs, kernel_1, [1, 1, 1, 1], padding='SAME')
      bias_1 = tf.nn.bias_add(conv_1, biases_1)
      hidden_1 = tf.nn.relu(bias_1, name=scope.name)
      util.activation_summary(hidden_1)

    with tf.variable_scope('layer-2') as scope:
      kernel_2 = util.variable_with_weight_decay(
          'weights', [3, 3, config.state_kernels, config.state_kernels])
      biases_2 = util.variable_on_cpu('biases', [config.state_kernels],
                                      tf.constant_initializer(0.1))
      conv_2 = tf.nn.conv2d(hidden_1, kernel_2, [1, 1, 1, 1], padding='SAME')
      bias_2 = tf.nn.bias_add(conv_2, biases_2)
      state_representation = tf.nn.relu(bias_2, name=scope.name)
      util.activation_summary(state_representation)

    return state_representation


def model_network(state, config, reuse):
  with tf.variable_scope('model', reuse=reuse):
    with tf.variable_scope('layer-1', reuse=reuse) as scope:
      kernel_1 = util.variable_with_weight_decay(
          'weights', [3, 3, config.state_kernels, config.state_kernels])
      biases_1 = util.variable_on_cpu('biases', [config.state_kernels],
                                      tf.constant_initializer(0.1))
      conv_1 = tf.nn.conv2d(state, kernel_1, [1, 1, 1, 1], padding='SAME')
      bias_1 = tf.nn.bias_add(conv_1, biases_1)
      normalized_1 = tf.contrib.layers.batch_norm(
          bias_1,
          decay=0.99,
          center=False,
          scale=False,
          is_training=config.is_training,
          scope=scope,
          reuse=reuse)
      hidden_layer_1 = tf.nn.relu(normalized_1, name=scope.name)
      util.activation_summary(hidden_layer_1)

    with tf.variable_scope('layer-2', reuse=reuse) as scope:
      kernel_2 = util.variable_with_weight_decay(
          'weights', [3, 3, config.state_kernels, config.state_kernels])
      biases_2 = util.variable_on_cpu('biases', [config.state_kernels],
                                      tf.constant_initializer(0.1))
      conv_2 = tf.nn.conv2d(
          hidden_layer_1, kernel_2, [1, 1, 1, 1], padding='SAME')
      bias_2 = tf.nn.bias_add(conv_2, biases_2)
      normalized_2 = tf.contrib.layers.batch_norm(
          bias_2,
          decay=0.99,
          center=False,
          scale=False,
          is_training=config.is_training,
          scope=scope,
          reuse=reuse)
      hidden_layer_2 = tf.nn.relu(normalized_2, name=scope.name)
      util.activation_summary(hidden_layer_2)

    with tf.variable_scope('layer-3', reuse=reuse) as scope:
      kernel_3 = util.variable_with_weight_decay(
          'weights', [3, 3, config.state_kernels, config.state_kernels])
      biases_3 = util.variable_on_cpu('biases', [config.state_kernels],
                                      tf.constant_initializer(0.1))
      conv_3 = tf.nn.conv2d(
          hidden_layer_2, kernel_3, [1, 1, 1, 1], padding='SAME')
      bias_3 = tf.nn.bias_add(conv_3, biases_3)
      normalized_3 = tf.contrib.layers.batch_norm(
          bias_3,
          decay=0.99,
          center=False,
          scale=False,
          is_training=config.is_training,
          scope=scope,
          reuse=reuse)
      next_state = tf.nn.relu(normalized_3, name=scope.name)

    return hidden_layer_1, next_state


def state_size(config):
  return config.input_height * config.input_width * config.state_kernels


def rollout_states(state, config):
  hidden_states = []
  states = [state]

  with tf.variable_scope('states'):
    for i in range(1, config.predictron_depth + 1):
      if config.shared_core:
        scope = 'shared-core'
        reuse = i > 1
      else:
        scope = 'core-%d' % i
        reuse = False

      with tf.variable_scope(scope, reuse=reuse):
        hidden_state, state = model_network(state, config, reuse)
        states.append(state)
        hidden_states.append(hidden_state)

    states = tf.reshape(
        tf.pack(states[:-1], 1),
        [config.batch_size, config.predictron_depth, state_size(config)])
    hidden_states = tf.reshape(
        tf.pack(hidden_states, 1),
        [config.batch_size, config.predictron_depth, state_size(config)])

    util.activation_summary(states)
    return states, hidden_states


def output_network(inputs, config):
  if config.shared_core:
    weights_1 = util.variable_with_weight_decay(
        'weights_1', [state_size(config), config.output_hidden_size])
    biases_1 = util.variable_on_cpu('biases_1', [config.output_hidden_size],
                                    tf.constant_initializer(0.0))
    weights_2 = util.variable_with_weight_decay(
        'weights_2', [config.output_hidden_size, config.reward_size])
    biases_2 = util.variable_on_cpu('biases_2', [config.reward_size],
                                    tf.constant_initializer(0.0))

    inputs = tf.reshape(
        inputs,
        [config.batch_size * config.predictron_depth, state_size(config)])
    hidden = tf.nn.xw_plus_b(inputs, weights_1, biases_1)
    logits = tf.nn.xw_plus_b(hidden, weights_2, biases_2)
  else:
    weights_1 = util.variable_with_weight_decay('weights_1', [
        1, config.predictron_depth, state_size(config),
        config.output_hidden_size
    ])
    weights_1 = tf.tile(weights_1,
                        [config.batch_size, 1, 1, 1])  # Manual broadcasting
    biases_1 = util.variable_on_cpu(
        'biases_1', [config.predictron_depth, 1, config.output_hidden_size],
        tf.constant_initializer(0.0))
    weights_2 = util.variable_with_weight_decay('weights_2', [
        1, config.predictron_depth, config.output_hidden_size,
        config.reward_size
    ])
    weights_2 = tf.tile(weights_2,
                        [config.batch_size, 1, 1, 1])  # Manual broadcasting
    biases_2 = util.variable_on_cpu(
        'biases_2', [config.predictron_depth, 1, config.reward_size],
        tf.constant_initializer(0.0))

    inputs = tf.reshape(
        inputs,
        [config.batch_size, config.predictron_depth, 1, state_size(config)])

    hidden = tf.batch_matmul(inputs, weights_1) + biases_1
    logits = tf.batch_matmul(hidden, weights_2) + biases_2

  return tf.reshape(
      logits, [config.batch_size, config.predictron_depth, config.reward_size])


def value_network(states, config):
  with tf.variable_scope('value') as scope:
    values = output_network(states, config)
    util.activation_summary(values)
    return values


def reward_network(hidden_states, config):
  with tf.variable_scope('reward') as scope:
    rewards = output_network(hidden_states, config)

    # Insert rewards[0] as zero
    rewards = tf.slice(
        rewards,
        begin=[0, 0, 0],
        size=[
            config.batch_size, config.predictron_depth - 1, config.reward_size
        ])
    rewards = tf.concat(
        concat_dim=1,
        values=[tf.zeros([config.batch_size, 1, config.reward_size]), rewards])

    util.activation_summary(rewards)
    return rewards


def discount_network(hidden_states, config):
  with tf.variable_scope('discount') as scope:
    logits = output_network(hidden_states, config)
    discounts = tf.nn.sigmoid(logits)

    # Insert discounts[0] as one
    discounts = tf.slice(
        discounts,
        begin=[0, 0, 0],
        size=[
            config.batch_size, config.predictron_depth - 1, config.reward_size
        ])
    discounts = tf.concat(
        concat_dim=1,
        values=[
            tf.ones([config.batch_size, 1, config.reward_size]), discounts
        ])

    util.activation_summary(discounts)
    return discounts


def lambda_network(hidden_states, config):
  with tf.variable_scope('lambda') as scope:
    logits = output_network(hidden_states, config)
    lambdas = tf.nn.sigmoid(logits, name=scope.name)

    # Set lambdas[-1] to zero
    lambdas = tf.slice(
        lambdas,
        begin=[0, 0, 0],
        size=[
            config.batch_size, config.predictron_depth - 1, config.reward_size
        ])
    lambdas = tf.concat(
        concat_dim=1,
        values=[lambdas, tf.zeros([config.batch_size, 1, config.reward_size])])

    util.activation_summary(lambdas)
    return lambdas


def preturn_network(rewards, discounts, values):
  # First reward must be zero, first discount must be one
  first_reward = tf.Assert(
      tf.reduce_all(tf.equal(rewards[:, 0, :], 0.0)), [rewards[:, 0, :]])
  first_discount = tf.Assert(
      tf.reduce_all(tf.equal(discounts[:, 0, :], 1.0)), [discounts[:, 0, :]])

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
  final_lambda = tf.Assert(
      tf.reduce_all(tf.equal(lambdas[:, -1, :], 0.0)), [lambdas[:, -1, :]])

  with tf.control_dependencies([final_lambda]):
    with tf.variable_scope('lambda_preturn'):
      accum_lambda = tf.cumprod(lambdas, axis=1, exclusive=True)
      lambda_bar = (1 - lambdas) * accum_lambda  # This should always sum to 1
      lambda_preturn = tf.reduce_sum(
          lambda_bar * preturns, reduction_indices=1)

      util.activation_summary(lambda_preturn)
      return lambda_preturn
