from flask import Flask
from flask import request

from models import bigru
from models import bigru2layers
from models import bigru2layers_dev

import os
import reader
import tensorflow as tf
import numpy as np


app = Flask(__name__)

relation_data_dir = '../../data/relation'
model_path = '../save_models/bigru2layers_dev_14421'
word_embedding_path = '../../../data/glove/glove.6B.300d.txt'
relation_embedding_path = '../../data/transE/relation_embeddings.txt'

os.environ['CUDA_VISIBLE_DEVICES'] = ''

id2tagsinwords_map = reader.buildID2TagsInWordsMap(relation_data_dir)
word_to_id, word_embedding = reader.load_vocabulary(word_embedding_path)
relation_embedding = reader.load_relation_embeddings(relation_embedding_path)
graph = bigru2layers_dev.BiGRU2LayersDev(len(word_embedding), len(relation_embedding),
                                         word_embedding, relation_embedding, batch=1)
#graph = bigru.BiGRU(len(word_embedding), len(relation_embedding),
#                    word_embedding, relation_embedding, batch=1)

#graph = bigru2layers.BiGRU2Layers(len(word_embedding), len(relation_embedding),
#                                  word_embedding, relation_embedding, batch=1)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()

#vars_to_train = tf.trainable_variables()
#vars_for_bn = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='relation_bn')
#vars_to_train = list(set(vars_to_train).union(set(vars_for_bn)))

#vars_all = tf.all_variables()
#vars_to_init = list(set(vars_all) - set(vars_to_train))
#init = tf.variables_initializer(vars_to_init)
#sess.run(init)

#saver = tf.train.Saver(vars_to_train)
saver.restore(sess, model_path)

f = open('../data/relative_rid.txt')
relative_rid = np.array([int(line.strip().split('\t')[0]) for line in f.readlines()])
f.close()
tag_vectors = np.zeros([1837, len(id2tagsinwords_map[0])], dtype=np.float32)
for i, t in enumerate(tag_vectors):
  t[:] = id2tagsinwords_map[relative_rid[i]]

@app.route('/relation_server', methods=['POST'])
def relation():
  if request.method == 'POST':
    qust = [word_to_id[t] if t in word_to_id else 0 for t in request.form['question'].strip().split(' ')]
    qust_idx = np.zeros([1, len(qust)], dtype=np.int32)
    qust_idx[0] = np.array(qust[:])
    lengths = np.array([qust_idx.shape[1]])
    #relation_candidates = np.arange(7523)
    feed = {graph.s: qust_idx, graph.lengths: lengths, graph.y_test: relative_rid, graph.training: False, graph.y_vector_test: tag_vectors}
    #ret = sess.run([graph.score], feed_dict=feed)
    #score = ret[0]
    #print score
    #sorted_idx = sorted(range(len(score)), key=lambda x: score[x], reverse=True)[:100]
    #return 'relation ranking: ' + ','.join(str(e) for e in sorted_idx)
    ret = sess.run([graph.z_online, graph.y_test_p, graph.score], feed_dict=feed)
    score = ret[2]
    pairs = zip(relative_rid, score)
    #sorted_idx = sorted(range(len(score)), key=lambda x: score[x], reverse=True)[:100]
    sorted_idx = sorted(range(len(pairs)), key=lambda x: pairs[x][1], reverse=True)[:200]
    return 'relation ranking: ' + ','.join(str(pairs[e][0]) for e in sorted_idx)
  return 'relation server!'
