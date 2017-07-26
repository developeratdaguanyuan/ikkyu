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


# ------------------ Load Relation Embedding and dictionary----------------
def load_relation_embeddings(path):
    file = open(path)
    lines = file.readlines()
    file.close()
    r = len(lines)
    c = len(lines[0].split())
    m = np.zeros(shape=(r, c))
    for line in lines:
        tokens = line.strip().split(':')
        values = tokens[1].strip().split(' ')
        m[int(tokens[0])] = [np.float32(values[i]) for i in range(0, len(values))]

    return m


# ----------------------------Data Producer--------------------------------
class DataProducer(object):
    def __init__(self, word_to_id, path, neg_num, cycle=True):
        self.questions, self.tags = list(), list()
        file = open(path)
        lines = file.readlines()
        file.close()

        for line in lines:
            tokens = line.strip().split('\t')
            question = [word_to_id[t] if t in word_to_id else 0 for t in tokens[0].strip().split(' ')]
            tag = int(tokens[1].strip())
            self.questions.append(question)
            self.tags.append(tag)

        self.neg_num = neg_num
        self.max_tag = max(self.tags) + 1
        self.cycle = cycle
        self.size = len(self.questions)
        self.cursor = 0

    def _generate_neg_samples(self, excluded_samples):
        neg_sample_list = list()
        for excluded_sample in excluded_samples:
            sample_list = [i for i in range(self.max_tag + 1)]
            sample_list.remove(excluded_sample)
            slice = random.sample(sample_list, self.neg_num)
            neg_sample_list.append(slice)
        return neg_sample_list

    def next(self, n):
        if (self.cursor + n - 1 >= self.size):
            self.cursor = 0
            if self.cycle is False:
                return None
        curr_questions = self.questions[self.cursor:self.cursor+n]
        curr_tags = self.tags[self.cursor:self.cursor+n]
        curr_neg_tags = self._generate_neg_samples(curr_tags)
        self.cursor += n

        lengths = [len(q) for q in curr_questions]
        max_length = max(l for l in lengths)

        question_list = np.zeros([n, max_length], dtype=np.int32)
        tag_list = np.array(curr_tags, dtype=np.int32)
        neg_tag_list = np.matrix(curr_neg_tags)

        for i, q in enumerate(question_list):
            q[:lengths[i]] = np.array(curr_questions[i][:lengths[i]])

        return question_list, tag_list, neg_tag_list


if __name__ == "__main__":
    relation_embedding_path = '../../data/transE/relation_embeddings.txt'
    m = load_relation_embeddings(relation_embedding_path)

    word_embedding_path = '../../../data/glove/glove.6B.300d.txt'
    word_to_id, word_embedding = load_vocabulary(word_embedding_path)

    producer = DataProducer(word_to_id, '../../data/relation/relation_train.txt', 9)
    questions, tags, neg_tags = producer.next(20)
    print 'tags:'
    print tags
    print
    print 'negative tags:'
    print neg_tags

