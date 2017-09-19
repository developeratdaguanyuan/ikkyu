import tensorflow as tf
import reader

from models import gru
from models import bigru
from models import bigru2layers
from models import bigru2layers_dev

relation_data_dir = '../../data/relation'
word_embedding_path = '../../../data/glove/glove.6B.300d.txt'
relation_embedding_path = '../../data/transE/relation_embeddings.txt'
test_path = "../../data/relation/relation_test.txt"

def main(_):
    id2tagsinwords_map = reader.buildID2TagsInWordsMap(relation_data_dir)
    word_to_id, word_embedding = reader.load_vocabulary(word_embedding_path)
    relation_embedding = reader.load_relation_embeddings(relation_embedding_path)
    test_data_producer = reader.DataProducer(id2tagsinwords_map, word_to_id, test_path, 1024, False)

    # BiGRU2LayersDev
    model_path = '../save_models/bigru2layers_dev_14421'
    graph = bigru2layers_dev.BiGRU2LayersDev(len(word_embedding), len(relation_embedding),
                                             word_embedding, relation_embedding, batch=1)
    print graph.evaluate(test_data_producer, model_path)

    # BiGRU2Layers
    # model_path = '../save_models/bigru2layers_9108'
    # graph = bigru2layers.BiGRU2Layers(len(word_embedding), len(relation_embedding),
    #                                   word_embedding, relation_embedding, batch=1)
    # print graph.evaluate(test_data_producer, model_path)

    # BiGRU
    # model_path = '../save_models_tmp/bigru_16698'
    #model_path = '../save_models/bigru_15939'
    #model_path = '../save_models/bigru_8349'
    #model_path = '../save_models/bigru_4554'
    #graph = bigru.BiGRU(len(word_embedding), len(relation_embedding),
    #                    word_embedding, relation_embedding, batch=1)
    #print graph.evaluate(test_data_producer, model_path)

'''
    # GRU
    model_path = '../save_models/gru_7590'
    graph = gru.GRU(len(word_embedding), len(relation_embedding),
                    word_embedding, relation_embedding)
    print graph.evaluate(test_data_producer, model_path)
'''

if __name__ == "__main__":
    tf.app.run()
