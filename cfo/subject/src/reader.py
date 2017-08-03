import csv
import os
import random
import numpy as np

# ------------------ Load Word Embedding and dictionary--------------------
def load_vocabulary(path):
    word_to_id = _load_vocabulary(path)
    word_embedding = _get_word_embedding(path, word_to_id)

    return word_to_id, word_embedding


def _load_vocabulary(path):
    word_to_id = dict()
    wv_file = open(path)
    while True:
        line = wv_file.readline()
        if not line:
            break
        tokens = line.strip().split()
        word_to_id[tokens[0]] = len(word_to_id) + 1 # Start from 1
    wv_file.close()

    return word_to_id


def _initialize_random_matrix(shape, scale=0.05, seed=0):
    if len(shape) != 2:
        raise ValueError("Shape of embedding matrix must be 2D, "
                         "got shape {}".format(shape))
    numpy_rng = np.random.RandomState(seed)

    return numpy_rng.uniform(low=-scale, high=scale, size=shape)


def _get_word_embedding(wv_path, word_to_id):
    m = _initialize_random_matrix((1 + len(word_to_id), 300))
    cnt = 0
    wv_file = open(wv_path)
    while True:
        line = wv_file.readline()
        if not line:
            break
        cnt += 1
        tokens = line.strip().split(' ')
        if word_to_id[tokens[0].strip()] != cnt:
            print('ERROR')
            break
        m[cnt] = [np.float32(tokens[i]) for i in range(1, len(tokens))]
    wv_file.close()

    return m

# ----------------------------Data Producer--------------------------------
class DataProducer(object):
    def __init__(self, word_to_id, path, cycle=True):
        self.questions, self.embeddings, self.neg_embeddings = list(), list(), list()
        file = open(path)
        lines = file.readlines()
        file.close()

        lastQ = ''
        question = list()
        embedding = list()
        idx = 1
        for line in lines:
            tokens = line.strip().split('\t')
            if lastQ == '' or tokens[0] != lastQ:
                if tokens[1] != '1':
                    print idx
                    print 'ERROR: line 71'
                    break
                lastQ = tokens[0]
                question = [word_to_id[t] if t in word_to_id else 0 for t in lastQ.strip().split(' ')]
                embedding = [float(value) for value in tokens[2].strip().split(' ')]
            else:
                if tokens[1] != '0':
                    print 'ERROR: line 78'
                    break
                neg_embeddings = [float(value) for value in tokens[2].strip().split(' ')]
                self.questions.append(question)
                self.embeddings.append(embedding)
                self.neg_embeddings.append(neg_embeddings)
            idx += 1

        if len(self.questions) != len(self.embeddings) \
            or len(self.questions) != len(self.neg_embeddings):
            print 'ERROR: line 87'

        self._shuffle()
        self.cycle = cycle
        self.size = len(self.questions)
        self.cursor = 0


    def _shuffle(self):
        tmp = zip(self.questions, self.embeddings, self.neg_embeddings)
        random.shuffle(tmp)
        self.questions, self.embeddings, self.neg_embeddings = zip(*tmp)

    def next(self, n):
        if (self.cursor + n - 1 >= self.size):
            self.cursor = 0
            if self.cycle is False:
                return None
            self._shuffle();
        curr_questions = self.questions[self.cursor:self.cursor+n]
        curr_embeddings = self.embeddings[self.cursor:self.cursor+n]
        curr_neg_embeddings = self.neg_embeddings[self.cursor:self.cursor+n]

        self.cursor += n

        lengths = [len(q) for q in curr_questions]
        max_length = max(l for l in lengths)

        question_list = np.zeros([n, max_length], dtype=np.int32)
        for i, q in enumerate(question_list):
            q[:lengths[i]] = np.array(curr_questions[i][:lengths[i]])
        embedding_list = np.matrix(curr_embeddings)
        neg_embedding_list = np.matrix(curr_neg_embeddings)

        return question_list, embedding_list, neg_embedding_list


if __name__ == "__main__":
    word_embedding_path = '../../../data/glove/glove.6B.300d.txt'
    word_to_id, word_embedding = load_vocabulary(word_embedding_path)

    producer = DataProducer(word_to_id, '../../data/subject/subject_test.txt')
    questions, embeddings, neg_embeddings = producer.next(20)
    print 'questions:'
    print questions
    print
    print 'embeddings:'
    print embeddings
    print
    print 'neg embeddings:'
    print neg_embeddings
