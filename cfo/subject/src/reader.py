import csv
import os
import random
import numpy as np
from random import shuffle


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
        self.questions, self.tags, self.entity_embeddings = list(), list(), list()
        file = open(path)
        lines = file.readlines()
        file.close()

        shuffle(lines)
        for line in lines:
            tokens = line.strip().split('\t')
            question = [word_to_id[t] if t in word_to_id else 0 for t in tokens[0].strip().split(' ')]
            tag = int(tokens[1].strip())
            embedding = [float(value) for value in tokens[2].strip().split(' ')]
            self.questions.append(question)
            self.tags.append(tag)
            self.entity_embeddings.append(embedding)

        self.cycle = cycle
        self.size = len(self.questions)
        self.cursor = 0

    def next(self, n):
        if (self.cursor + n - 1 >= self.size):
            self.cursor = 0
            if self.cycle is False:
                return None
        curr_questions = self.questions[self.cursor:self.cursor+n]
        curr_tags = self.tags[self.cursor:self.cursor+n]
        curr_entity_embeddings = self.entity_embeddings[self.cursor:self.cursor+n]
        self.cursor += n

        lengths = [len(q) for q in curr_questions]
        max_length = max(l for l in lengths)

        question_list = np.zeros([n, max_length], dtype=np.int32)
        tag_list = np.array(curr_tags, dtype=np.float32)
        embedding_list = np.matrix(curr_entity_embeddings)
        for i, q in enumerate(question_list):
            q[:lengths[i]] = np.array(curr_questions[i][:lengths[i]])

        return question_list, tag_list, embedding_list


if __name__ == "__main__":
    word_embedding_path = '../../../data/glove/glove.6B.300d.txt'
    word_to_id, word_embedding = load_vocabulary(word_embedding_path)

    producer = DataProducer(word_to_id, '../../data/subject/subject_test.txt')
    questions, tags, embeddings = producer.next(20)
    print 'questions:'
    print questions
    print
    print 'tags:'
    print tags
    print
    print 'embeddings:'
    print embeddings
