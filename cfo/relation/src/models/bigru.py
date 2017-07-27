import tensorflow as tf
import numpy as np
import logging


class BiGRU(object):
  def __init__(self, vocab_size, relation_size, word_vectors, relation_vectors,
               batch=100, neg_sample_num=1024,
               relation_dim=256, embedding_dim=300, hidden_dim=300,
               learning_rate=1e-3, training=True):
    self.batch = batch
    self.neg_sample_num = neg_sample_num

    self.relation_dim = relation_dim
    self.embedding_dim = embedding_dim
    self.hidden_dim = hidden_dim
    self.lr = learning_rate
    self.vocab_sz = vocab_size
    self.relation_sz = relation_size

    # Placeholders
    # Sentence placeholders with shape (batch, time_steps)
    self.s = tf.placeholder(tf.int32, [self.batch, None])
    # Sentence length placeholders with shape (batch)
    self.lengths = tf.placeholder(tf.int32, [self.batch])
    # Label placeholders with shape (batch, 1)
    self.y = tf.placeholder(tf.int32, [self.batch])
    self.y_neg = tf.placeholder(tf.int32, [self.batch, self.neg_sample_num])

    # Embedding layer
    self.embeddings = tf.get_variable('embedding_matrix', dtype='float',
                                      initializer=tf.constant_initializer(word_vectors),
                                      shape=[self.vocab_sz, self.embedding_dim], trainable=False)
    self.rel_embeddings = tf.get_variable('rel_embedding_matrix', dtype='float',
                                          initializer=tf.constant_initializer(relation_vectors),
                                          shape=[self.relation_sz, self.relation_dim], trainable=False)

    # Lookup
    e = tf.nn.embedding_lookup(self.embeddings, self.s)
    y_pos_e = tf.nn.embedding_lookup(self.rel_embeddings, self.y)
    y_neg_e = tf.nn.embedding_lookup(self.rel_embeddings, self.y_neg)

    # Project Layer
    p = self._apply_relu(e, self.embedding_dim, self.embedding_dim, 'embedding_project')
    y_pos_p = self._apply_relu(y_pos_e, self.relation_dim, self.hidden_dim, 'relation_project')
    y_neg_p = self._apply_relu(y_neg_e, self.relation_dim, self.hidden_dim, 'relation_project', True)

    # BiGRU Layer on p
    z, _ = self._apply_bigru(p, self.lengths, self.hidden_dim, 'encoder')
    z = tf.transpose(z, perm=[1, 0, 2])[-1]
    z = self._apply_linear(z, self.hidden_dim * 2, self.hidden_dim, 'bigru_output_project')

    # Merge Layer
    z_expansion = tf.reshape(tf.tile(tf.expand_dims(z, 1), [1, self.neg_sample_num, 1]), [-1, self.hidden_dim])
    y_neg_p = tf.reshape(y_neg_p, [-1, self.hidden_dim])
    score_pos = self._apply_dot(z, y_pos_p)
    score_pos = tf.reshape(tf.tile(tf.expand_dims(score_pos, 1), [1, self.neg_sample_num, 1]), [-1, 1])
    score_neg = self._apply_dot(z_expansion, y_neg_p)
    self.loss = tf.reduce_mean(tf.maximum(0., 1. - score_pos + score_neg))
    self.rank = tf.count_nonzero(tf.maximum(0., score_neg - score_pos))
    self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss)


  def _apply_dot(self, input_1, input_2):
    tmp = tf.multiply(input_1, input_2)
    return tf.reduce_sum(tmp, 1, keep_dims=True)

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


  def train(self, train_data_producer, valid_data_producer, num_epoch):
    logging.basicConfig(filename='../logs/bigru.log', filemode='w', level=logging.INFO)
    saver = tf.train.Saver()
    with tf.Session() as self.sess:
      self.sess.run(tf.global_variables_initializer())

      train_loss, valid_loss, valid_rank, rank = 0, float('inf'), float('inf'), 0
      num_iteration = num_epoch * train_data_producer.size / self.batch
      for i in range(num_iteration):
        questions, tags, neg_tags = train_data_producer.next(self.batch)
        lengths = np.array([questions.shape[1] for _ in range(self.batch)])

        feed = {self.s: questions, self.lengths: lengths, self.y: tags, self.y_neg: neg_tags}
        ret = self.sess.run([self.optimizer, self.loss, self.rank], feed_dict=feed)
        train_loss += ret[1]
        rank += ret[2]

        if i > 0 and i % (train_data_producer.size / self.batch) == 0:
          # train info
          train_loss = train_loss / train_data_producer.size
          rank = float(rank) / train_data_producer.size
          logging.info("[train loss] %5.5f [train rank] %5.5f", train_loss, rank)
          train_loss, rank = 0, 0

          # valid info
          valid_loss_t, valid_rank_t = self._evaluate(valid_data_producer)
          logging.info("[valid loss] %5.5f [valid rank] %5.5f", valid_loss_t, valid_rank_t)

          # write model
          # if valid_loss_t <= valid_loss:
          #  valid_loss = valid_loss_t
          #  save_path = saver.save(self.sess, "../save_models/bigru_" + str(i))
          #  logging.info("Model saved in file: %s", save_path)
          if valid_rank_t <= valid_rank:
            valid_rank = valid_rank_t
            save_path = saver.save(self.sess, "../save_models/bigru_" + str(i))
            logging.info("Model saved in file: %s", save_path)
            

  def _evaluate(self, data_producer):
    '''
    Do Evaluation during training
    :param data_producer: data producer
    :return: log loss
    '''
    loss, rank = 0, 0
    while True:
      data = data_producer.next(self.batch)
      if data is None:
        loss = loss / data_producer.size
        rank = float(rank) / data_producer.size
        break
      questions, tags, neg_tags = data
      lengths = np.array([questions.shape[1] for _ in range(self.batch)])
      feed = {self.s: questions, self.lengths: lengths, self.y: tags, self.y_neg: neg_tags}
      ret = self.sess.run([self.loss, self.rank], feed_dict=feed)
      loss += ret[0]
      rank += ret[1]

    return loss, rank


  def evaluate(self, data_producer, model_path):
    '''
    Do Evaluation
    :param data_producer: data producer
    :param model_path: model path
    :return: log loss
    '''
    with tf.Session() as self.sess:
      self.sess.run(tf.global_variables_initializer())
      saver = tf.train.Saver()
      saver.restore(self.sess, model_path)
      return self._evaluate(data_producer)

    return None
