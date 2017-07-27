import tensorflow as tf
import reader

from models import gru
from models import bigru

word_embedding_path = '../../../data/glove/glove.6B.300d.txt'
test_path = "../../data/focus/focus_test.txt"

def main(_):
    word_to_id, word_embedding = reader.load_vocabulary(word_embedding_path)
    test_data_producer = reader.DataProducer(word_to_id, test_path, False)

    # GRU
    # model_path = '../save_models/gru_52575'
    # graph = gru.GRU(len(word_embedding), 2, word_embedding)
    # print graph.evaluate(test_data_producer, model_path)

    # BiGRU
    model_path = '../save_models/bigru_5608'
    graph = bigru.BiGRU(len(word_embedding), 2, word_embedding)
    print graph.evaluate(test_data_producer, model_path)

if __name__ == "__main__":
    tf.app.run()
