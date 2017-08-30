import tensorflow as tf
import reader

from models import gru
from models import bigru_crf
from models import bigru2layers
from models import bigru2layers_crf

word_embedding_path = '../../../data/glove/glove.6B.300d.txt'
train_path = "../../data/focus/focus_train.txt"
valid_path = "../../data/focus/focus_valid.txt"

def main(_):
    word_to_id, word_embedding = reader.load_vocabulary(word_embedding_path)

    train_data_producer = reader.DataProducer(word_to_id, train_path)
    valid_data_producer = reader.DataProducer(word_to_id, valid_path, False)

    graph = bigru2layers_crf.BiGRU2LayersCRF(len(word_embedding), 2, word_embedding)
    graph.train(train_data_producer, valid_data_producer, 100)

    #graph = bigru2layers.BiGRU2Layers(len(word_embedding), 2, word_embedding)
    #graph.train(train_data_producer, valid_data_producer, 100)

    #graph = bigru_cfo.BiGRUCRF(len(word_embedding), 2, word_embedding)
    #graph.train(train_data_producer, valid_data_producer, 100)

    # graph = gru.GRU(len(word_embedding), 2, word_embedding)
    # graph.train(train_data_producer, valid_data_producer, 100)


if __name__ == "__main__":
    tf.app.run()
