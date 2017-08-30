from flask import Flask
from flask import request

import os
import reader
import logging

import tensorflow as tf
import numpy as np

from models import bigru_crf
from models import bigru2layers_crf

app = Flask(__name__)

logging.basicConfig(filename='../logs/focus_server.log', filemode='w', level=logging.INFO)

word_embedding_path = '../../../data/glove/glove.6B.300d.txt'
word_to_id, word_embedding = reader.load_vocabulary(word_embedding_path)
model_path = '../save_models/bigru2layers_crf_50472'

os.environ['CUDA_VISIBLE_DEVICES'] = ''
graph = bigru2layers_crf.BiGRU2LayersCRF(len(word_embedding), 2, word_embedding, batch=1)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(sess, model_path)


@app.route('/focus_server', methods=['POST'])
def focus():
  if request.method == 'POST':
    qust = [word_to_id[t] if t in word_to_id else 0 for t in request.form['question'].strip().split(' ')]
    qust_idx = np.zeros([1, len(qust)], dtype=np.int32)
    qust_idx[0] = np.array(qust[:])
    lengths = np.array([qust_idx.shape[1]])

    feed = {graph.words: qust_idx, graph.lengths: lengths, graph.training: False}
    logits, transition = sess.run([graph.logits, graph.transition], feed_dict=feed)
    viterbi_sequence, _ = tf.contrib.crf.viterbi_decode(logits[0], transition)

    result = 'focus tagging: ' + ','.join(str(e) for e in viterbi_sequence)
    logging.info("[question] %s", request.form['question'].strip())
    logging.info("[taggings] %s", result)
    return result
  return 'focus!'
