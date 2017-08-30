import tensorflow as tf
import reader

from models import gru
from models import bigru
from models import bigru_crf
from models import bigru2layers_crf

word_embedding_path = '../../../data/glove/glove.6B.300d.txt'
test_path = "../../data/focus/focus_test.txt"

def main(_):
    word_to_id, word_embedding = reader.load_vocabulary(word_embedding_path)
    test_data_producer = reader.DataProducer(word_to_id, test_path, False)

    model_path = '../save_models/bigru2layers_crf_50472'
    graph = bigru2layers_crf.BiGRU2LayersCRF(len(word_embedding), 2, word_embedding, batch=1)
    print graph.evaluate(test_data_producer, model_path)

    # model_path = '../save_models/bigru_cfo_40658'
    # graph = bigru_cfo.BiGRUCFO(len(word_embedding), 2, word_embedding, batch=1)
    # print graph.evaluate(test_data_producer, model_path)

    # GRU
    # model_path = '../save_models/gru_52575'
    # graph = gru.GRU(len(word_embedding), 2, word_embedding)
    # print graph.evaluate(test_data_producer, model_path)

    # BiGRU
    #model_path = '../save_models/bigru_5608'
    #model_path = '../save_models/bigru_26638'
    #graph = bigru.BiGRU(len(word_embedding), 2, word_embedding)
    #print graph.evaluate(test_data_producer, model_path)

if __name__ == "__main__":
    tf.app.run()
