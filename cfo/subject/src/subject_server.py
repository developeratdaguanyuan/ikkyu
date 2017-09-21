from flask import Flask
from flask import request

import os
import reader
import logging
import redis

import tensorflow as tf
import numpy as np

from models import bigru

app = Flask(__name__)
os.environ['CUDA_VISIBLE_DEVICES'] = ''
logging.basicConfig(filename='../logs/subject_server.log', filemode='w', level=logging.INFO)

word_embedding_path = '../../../data/glove/glove.6B.300d.txt'
model_path = '../save_models/bigru_69384'

word_to_id, word_embedding = reader.load_vocabulary(word_embedding_path)

graph = bigru.BiGRU(len(word_embedding), word_embedding, batch=1, training=False)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(sess, model_path)

r = redis.StrictRedis(host='localhost', port=6379, db=3)

@app.route('/subject_server', methods=['POST'])
def focus():
  if request.method == 'POST':
    qust = [word_to_id[t] if t in word_to_id else 0 for t in request.form['question'].strip().split(' ')]
    qust_idx = np.zeros([1, len(qust)], dtype=np.int32)
    qust_idx[0] = np.array(qust[:])
    lengths = np.array([qust_idx.shape[1]])

    subject_ids = [sid for sid in request.form['subject_ids'].strip().split(',')]
    tag_vectors = np.zeros([len(subject_ids), 256], dtype=np.float32)
    for i, t in enumerate(tag_vectors):
      t[:] = [float(v) for v in r.get(subject_ids[i]).strip().split(' ')]

    feed = {graph.s: qust_idx, graph.lengths: lengths, graph.test_tags: tag_vectors}
    scores = sess.run([graph.test_score], feed_dict=feed)[0]
    scores_str = ','.join(str(e) for e in scores)

    pairs = zip(subject_ids, scores)
    sorted_idx = sorted(range(len(pairs)), key=lambda x: pairs[x][1], reverse=True)
    id_list = ','.join(str(pairs[e][0]) for e in sorted_idx)
    score_list = ','.join(str(pairs[e][1]) for e in sorted_idx)

    logging.info("[question] %s", request.form['question'].strip())
    logging.info("[id list] %s", id_list)
    logging.info("[scores] %s", score_list)

    return 'subject ranking: ' + id_list
  return 'focus!'
