import tensorflow as tf
import logging


class GRU(object):
  def __init__(self, vocab_size, class_size, word_vectors,
               batch=100, embedding_dim=300, hidden_dim=300, learning_rate=1e-3,
               training=True):
    self.batch = batch

    self.embedding_dim = embedding_dim
    self.hidden_dim = hidden_dim
    self.lr = learning_rate
    self.vocab_sz = vocab_size
    self.class_sz = class_size

    # Placeholders
    self.words = tf.placeholder(tf.int32, [self.batch, None])
    self.tags = tf.placeholder(tf.int32, [self.batch, None])

    # Embedding layer
    # Tensor e{1, 2} have shape (batch, time_steps, embedding_dim)
    self.embeddings = tf.get_variable('embedding_matrix', dtype='float',
                                      initializer=tf.constant_initializer(word_vectors),
                                      shape=[self.vocab_sz, self.embedding_dim], trainable=False)
    word_embeddings = tf.nn.embedding_lookup(self.embeddings, self.words)

    # Project Layer
    word_projected = self._apply_relu(word_embeddings, self.embedding_dim,
                                      self.embedding_dim, 'embedding_project')
    # GRU
    z, _ = self._apply_gru(word_projected, self.hidden_dim, 'encoder')

    #if training:
    #  z = tf.nn.dropout(z, 0.85)

    # Project Layer
    logits = self._apply_linear(z, self.hidden_dim, self.class_sz)
    self.sequence_tags = tf.to_int32(tf.argmax(logits, 2))

    # Loss & Optimizer
    all_ones = tf.ones([self.batch, tf.shape(self.words)[1]])
    self.loss = tf.contrib.seq2seq.sequence_loss(logits, self.tags, all_ones)
    self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    # Accuracy OR Error
    self.error = tf.reduce_sum(tf.abs(tf.to_int32(tf.argmax(logits, 2)) - self.tags))

  def _apply_relu(self, inputs, input_unit_num, output_unit_num, scope=None, reuse=False):
    rank = len(inputs.get_shape())
    if rank == 3:
      time_steps = tf.shape(inputs)[1]
      inputs_2d = tf.reshape(inputs, [-1, input_unit_num])
    else:
      inputs_2d = inputs

    scope_name = scope or 'relu'
    with tf.variable_scope(scope_name, reuse=reuse):
      initializer = tf.random_uniform_initializer(-0.1, 0.1)
      W = tf.get_variable('W', [input_unit_num, output_unit_num], initializer=initializer)
      b = tf.get_variable('b', [output_unit_num], initializer=initializer)
      z = tf.nn.relu(tf.matmul(inputs_2d, W) + b)
    if rank == 3:
      output_shape = tf.stack([-1, time_steps, output_unit_num])
      return tf.reshape(z, output_shape)

    return z


  def _apply_gru(self, inputs, unit_num, scope=None, reuse=False):
    '''
    GRU model
    :param input: word vectors of a sentence with shape (batch, length, embedding_dim)
    :param unit_num: number of unit in LSTM
    :param reuse: reuse layers
    :return:
    '''
    scope_name = scope or 'gru'
    with tf.variable_scope(scope_name, reuse=reuse):
      initializer = tf.random_uniform_initializer(-0.1, 0.1)
      cell = tf.nn.rnn_cell.GRUCell(unit_num)
      z, final_state = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32)

    return z, final_state


  def _apply_linear(self, inputs, input_unit_num, output_unit_num, scope=None, reuse=False):
    rank = len(inputs.get_shape())
    if rank == 3:
      time_steps = tf.shape(inputs)[1]
      inputs_2d = tf.reshape(inputs, [-1, input_unit_num])
    else:
      inputs_2d = inputs

    scope_name = scope or 'linear'
    with tf.variable_scope(scope_name, reuse=reuse):
      initializer = tf.random_uniform_initializer(-0.1, 0.1)
      W = tf.get_variable('W', [input_unit_num, output_unit_num], initializer=initializer)
      z = tf.matmul(inputs_2d, W)
    if rank == 3:
      output_shape = tf.stack([-1, time_steps, output_unit_num])
      return tf.reshape(z, output_shape)

    return z


  def train(self, train_data_producer, valid_data_producer, num_epoch):
    logging.basicConfig(filename='../logs/gru.log', filemode='w', level=logging.INFO)
    saver = tf.train.Saver()
    with tf.Session() as self.sess:
      self.sess.run(tf.global_variables_initializer())

      train_loss, min_valid_loss = 0, float('inf')
      train_error, min_valid_error = 0, 0
      num_iteration = num_epoch * train_data_producer.size / self.batch
      for i in range(num_iteration):
        words, tags = train_data_producer.next(self.batch)
        feed = {self.words: words, self.tags: tags}

        ret = self.sess.run([self.optimizer, self.loss, self.error], feed_dict=feed)
        train_loss += ret[1]
        train_error += ret[2]

        if i > 0 and i % (train_data_producer.size / self.batch) == 0:
          # train info
          train_loss = train_loss / train_data_producer.size * self.batch
          logging.info("[train loss] %5.3f [train error] %d", train_loss, train_error)
          train_loss, train_error = 0, 0

          # valid info
          valid_loss, valid_error = self._evaluate(valid_data_producer)
          logging.info("[valid loss] %5.3f [valid error] %d", valid_loss, valid_error)

          # write model
          if valid_loss <= min_valid_loss:
            min_valid_loss = valid_loss
            save_path = saver.save(self.sess, "../save_models/gru_" + str(i))
            logging.info("Model saved in file: %s", save_path)


  def _evaluate(self, data_producer):
    loss_t, error_t = 0, 0
    while True:
      data = data_producer.next(self.batch)
      if data is None:
        loss_t = loss_t / data_producer.size * self.batch
        break
      words, tags = data
      feed = {self.words: words, self.tags: tags}
      ret = self.sess.run([self.loss, self.error], feed_dict=feed)
      loss_t += ret[0]
      error_t += ret[1]

    return loss_t, error_t


  def evaluate(self, data_producer, model_path):
    with tf.Session() as self.sess:
      self.sess.run(tf.global_variables_initializer())
      saver = tf.train.Saver()
      saver.restore(self.sess, model_path)
      return self._evaluate(data_producer)

    return None
