import sys
import logging

import numpy as np
import tensorflow as tf


class BiGRU2LayersCRF(object):
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
    self.lengths = tf.placeholder(tf.int32, [self.batch])
    self.training = tf.placeholder(tf.bool, name='training')

    # Embedding layer
    self.embeddings = tf.get_variable('embedding_matrix', dtype='float',
                                      initializer=tf.constant_initializer(word_vectors),
                                      shape=[self.vocab_sz, self.embedding_dim], trainable=False)
    word_embeddings = tf.nn.embedding_lookup(self.embeddings, self.words)

    # Project Layer
    word_projected = self._apply_relu(word_embeddings, self.embedding_dim,
                                      self.hidden_dim, 'embedding_project')

    # BiGRU Layers
    z, _ = self._apply_bigru(word_projected, self.lengths, self.hidden_dim, '1st_encoder')
    z = tf.cond(self.training, lambda: tf.nn.dropout(z, 0.5), lambda: z)
    z, _ = self._apply_bigru(z, self.lengths, self.hidden_dim * 2, '2nd_encoder')

    # Project Layer
    self.logits = self._apply_linear(z, self.hidden_dim * 4, self.class_sz, 'gru2class_projection')

    # CRF
    self.transition = tf.get_variable("transitions", [self.class_sz, self.class_sz])
    log_likelihood, _ = tf.contrib.crf.crf_log_likelihood(self.logits, self.tags,
                                                          self.lengths, transition_params=self.transition)

    # Loss & Optimizer
    self.loss = tf.reduce_mean(-log_likelihood)
    self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss)


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


  def _apply_bigru(self, inputs, lengths, unit_num, scope=None, reuse=False):
    scope_name = scope or 'bigru'
    with tf.variable_scope(scope_name, reuse=reuse):
      initializer = tf.random_uniform_initializer(-0.1, 0.1)
      cell_fw = tf.nn.rnn_cell.GRUCell(unit_num)
      cell_bw = tf.nn.rnn_cell.GRUCell(unit_num)
      init_state_fw = tf.get_variable('init_state_fw', [1, unit_num], initializer=initializer)
      init_state_fw = tf.tile(init_state_fw, [self.batch, 1])
      init_state_bw = tf.get_variable('init_state_bw', [1, unit_num], initializer=initializer)
      init_state_bw = tf.tile(init_state_bw, [self.batch, 1])
      z, final_state = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, inputs, lengths,
                                                       initial_state_fw=init_state_fw,
                                                       initial_state_bw=init_state_bw)
      z_fw, z_bw = z
      z = tf.concat([z_fw, z_bw], 2)

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
    logging.basicConfig(filename='../logs/bigru2layers_crf.log', filemode='w', level=logging.INFO)
    saver = tf.train.Saver()
    with tf.Session() as self.sess:
      self.sess.run(tf.global_variables_initializer())

      train_loss, min_valid_loss, min_valid_error = 0, float('inf'), sys.maxint
      num_iteration = num_epoch * train_data_producer.size / self.batch
      for i in range(num_iteration):
        words, tags = train_data_producer.next(self.batch)
        lengths = np.array([words.shape[1] for _ in range(self.batch)])
        feed = {self.words: words, self.tags: tags, self.lengths: lengths, self.training: True}
        train_loss += self.sess.run([self.optimizer, self.loss], feed_dict=feed)[1]

        if i > 0 and i % (train_data_producer.size / self.batch) == 0:
          # train info
          train_loss = train_loss / train_data_producer.size * self.batch
          logging.info("[train loss] %5.3f", train_loss)
          train_loss = 0

          # valid info
          valid_loss, valid_error = self._evaluate(valid_data_producer)
          logging.info("[valid loss] %5.3f [valid error] %d", valid_loss, valid_error)

          # write model
          if valid_loss <= min_valid_loss or valid_error <= min_valid_error:
            min_valid_loss = valid_loss if valid_loss <= min_valid_loss else min_valid_loss
            min_valid_error = valid_error if valid_error <= min_valid_error else min_valid_error
            save_path = saver.save(self.sess, "../save_models/bigru2layers_crf_" + str(i))
            logging.info("Model saved in file: %s", save_path)


  def _evaluate(self, data_producer):
    loss_t, error_t = 0, 0
    while True:
      data = data_producer.next(self.batch)
      if data is None:
        loss_t = loss_t / data_producer.size * self.batch
        break
      words, tags = data
      lengths = np.array([words.shape[1] for _ in range(self.batch)])
      feed = {self.words: words, self.tags: tags, self.lengths: lengths, self.training: False}
      logits, transition, loss = self.sess.run([self.logits, self.transition, self.loss], feed_dict=feed)
      loss_t += loss
      for logit, tag in zip(logits, tags):
        viterbi_sequence, _ = tf.contrib.crf.viterbi_decode(logit, transition)
        error_t += 0 if viterbi_sequence==tag.tolist() else 1

    return loss_t, error_t

  def evaluate(self, data_producer, model_path):
    with tf.Session() as self.sess:
      self.sess.run(tf.global_variables_initializer())
      saver = tf.train.Saver()
      saver.restore(self.sess, model_path)
      return self._evaluate(data_producer)

    return None
