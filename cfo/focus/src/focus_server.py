from flask import Flask
from flask import request

import tensorflow as tf
import numpy as np
from models import bigru
import reader


app = Flask(__name__)

word_embedding_path = '../../../data/glove/glove.6B.300d.txt'
word_to_id, word_embedding = reader.load_vocabulary(word_embedding_path)
model_path = '../save_models/bigru_4206'

graph = bigru.BiGRU(len(word_embedding), 2, word_embedding, batch=1)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(sess, model_path)


@app.route('/focus_server', methods=['POST'])
def focus():
  if request.method == 'POST':
    qust = [word_to_id[t] if t in word_to_id else 0 for t in request.form['qust'].strip().split(' ')]
    qust_idx = np.zeros([1, len(qust)], dtype=np.int32)
    qust_idx[0] = np.array(qust[:])
    lengths = np.array([qust_idx.shape[1]])

    #all_ones = np.ones([1, len(qust)], dtype=np.int32)
    feed = {graph.words: qust_idx, graph.lengths: lengths}
    ret = sess.run([graph.sequence_tags], feed_dict=feed)
    print ret[0]
    return 'Got your question: ' + ','.join(str(e) for e in qust) + '\n'
  return 'focus!'
