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


# -------------------Load Relation Embedding and dictionary----------------
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


# ------------------Build id2tagsinwords_map-------------------------------
def _loadRelationVocabulary(directory):
    vocab = dict()
    files = os.listdir(directory)
    for file in files:
        if not os.path.isdir(directory + "/" + file):
            f = open(directory + "/" + file)
            iter_f = iter(f)
            for line in iter_f:
                tokens = line.strip().split('\t')[2].strip().split(' ')
                for token in tokens:
                    if token not in vocab:
                        vocab[token] = len(vocab)
    return vocab

def buildID2TagsInWordsMap(directory):
    rel_vocab = _loadRelationVocabulary(directory)
    length = len(rel_vocab)
    id2tagsinwords_map = dict()
    files = os.listdir(directory)
    for file in files:
        if os.path.isdir(directory + "/" + file):
            continue
        f = open(directory + "/" + file)
        iter_f = iter(f)
        for line in iter_f:
            parts = line.strip().split('\t')
            tokens = parts[2].strip().split(' ')
            tag_in_words = [rel_vocab[w] for w in tokens]
            vector = np.zeros([length], dtype=np.float32)
            for i in tag_in_words:
                vector[i] = 1
            if int(parts[1]) in id2tagsinwords_map:
                assert np.array_equal(id2tagsinwords_map[int(parts[1])], vector), 'wrong in id2tagsinwords_map'
            id2tagsinwords_map[int(parts[1])] = vector

    return id2tagsinwords_map


# ----------------------------Data Producer--------------------------------
class DataProducer(object):
    def __init__(self, id2tagsinwords_map, word_to_id, path, neg_num, cycle=True):
        self.questions, self.tags, self.tags_in_words = list(), list(), list()
        file = open(path)
        lines = file.readlines()
        file.close()

        for line in lines:
            tokens = line.strip().split('\t')
            question = [word_to_id[t] if t in word_to_id else 0 for t in tokens[0].strip().split(' ')]
            tag = int(tokens[1].strip())
            self.questions.append(question)
            self.tags.append(tag)

        self.id2tagsinwords_map = id2tagsinwords_map
        self.neg_num = neg_num
        self.max_tag = max(self.tags) + 1
        self.cycle = cycle
        self.size = len(self.questions)
        self.cursor = 0

        f = open('../data/relative_rid.txt')
        self.rids = [int(rid.strip()) for rid in f.readlines()]
        f.close()

    def _generate_neg_samples(self, excluded_samples):
        neg_sample_list = list()
        for excluded_sample in excluded_samples:
            sample_list = self.rids[:]
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

        lengths = [len(q) if len(q) < 30 else 30 for q in curr_questions]
        max_length = max(l for l in lengths)

        tag_list = np.array(curr_tags, dtype=np.int32)
        neg_tag_list = np.matrix(curr_neg_tags)
        question_list = np.zeros([n, max_length], dtype=np.int32)
        for i, q in enumerate(question_list):
            q[:lengths[i]] = np.array(curr_questions[i][:lengths[i]])
        tag_vector = np.zeros([n, len(self.id2tagsinwords_map[0])], dtype=np.float32)
        for i, t in enumerate(tag_vector):
            t[:] = self.id2tagsinwords_map[curr_tags[i]]
        neg_tag_vector = np.zeros([n, self.neg_num, len(self.id2tagsinwords_map[0])], dtype=np.float32)
        for i, t in enumerate(neg_tag_vector):
            for j, s in enumerate(t):
                s[:] = self.id2tagsinwords_map[curr_neg_tags[i][j]]
        return question_list, tag_list, neg_tag_list, tag_vector, neg_tag_vector


if __name__ == "__main__":
    np.set_printoptions(threshold=np.nan)
    id2tagsinwords_map = buildID2TagsInWordsMap('../../data/relation')

    relation_embedding_path = '../../data/transE/relation_embeddings.txt'
    m = load_relation_embeddings(relation_embedding_path)

    word_embedding_path = '../../../data/glove/glove.6B.300d.txt'
    word_to_id, word_embedding = load_vocabulary(word_embedding_path)

    producer = DataProducer(id2tagsinwords_map, word_to_id, '../../data/relation/relation_train.txt', 3)
    questions, tags, neg_tags, tag_vectors, neg_tag_vectors = producer.next(20)
    print 'tags:'
    print tags
    print
    print 'negative tags:'
    print neg_tags
    print
    print 'tag vectors:'
    print tag_vectors
    print
    print 'neg tag vectors:'
    print neg_tag_vectors
