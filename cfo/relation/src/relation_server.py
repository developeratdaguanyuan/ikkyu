from flask import Flask
from flask import request

import os
import tensorflow as tf
import numpy as np
from models import bigru
import reader


app = Flask(__name__)

word_embedding_path = '../../../data/glove/glove.6B.300d.txt'
relation_embedding_path = '../../data/transE/relation_embeddings.txt'
word_to_id, word_embedding = reader.load_vocabulary(word_embedding_path)
relation_embedding = reader.load_relation_embeddings(relation_embedding_path)
model_path = '../save_models/bigru_16698'

os.environ['CUDA_VISIBLE_DEVICES'] = ''
word_to_id, word_embedding = reader.load_vocabulary(word_embedding_path)
relation_embedding = reader.load_relation_embeddings(relation_embedding_path)

graph = bigru.BiGRU(len(word_embedding), len(relation_embedding),
                    word_embedding, relation_embedding, batch=1)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(sess, model_path)

@app.route('/relation_server', methods=['POST'])
def relation():
  if request.method == 'POST':
    qust = [word_to_id[t] if t in word_to_id else 0 for t in request.form['qust'].strip().split(' ')]
    qust_idx = np.zeros([1, len(qust)], dtype=np.int32)
    qust_idx[0] = np.array(qust[:])
    lengths = np.array([qust_idx.shape[1]])

    feed = {graph.s: qust_idx, graph.lengths: lengths}
    ret = sess.run([graph.score], feed_dict=feed)
    score = ret[0]
    sorted_idx = sorted(range(len(score)), key=lambda x: score[x], reverse=True)[:100]

    return 'Got your question: ' + ','.join(str(e) for e in sorted_idx) + '\n'
  return 'relation server!'
