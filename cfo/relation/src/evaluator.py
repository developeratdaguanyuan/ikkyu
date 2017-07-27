import tensorflow as tf
import reader

from models import gru
from models import bigru

word_embedding_path = '../../../data/glove/glove.6B.300d.txt'
relation_embedding_path = '../../data/transE/relation_embeddings.txt'
test_path = "../../data/relation/relation_test.txt"

def main(_):
    word_to_id, word_embedding = reader.load_vocabulary(word_embedding_path)
    relation_embedding = reader.load_relation_embeddings(relation_embedding_path)
    test_data_producer = reader.DataProducer(word_to_id, test_path, 1024, False)

    # BiGRU
    model_path = '../save_models/bigru_9108'
    graph = bigru.BiGRU(len(word_embedding), len(relation_embedding),
                        word_embedding, relation_embedding)
    print graph.evaluate(test_data_producer, model_path)

'''
    # GRU
    model_path = '../save_models/gru_7590'
    graph = gru.GRU(len(word_embedding), len(relation_embedding),
                    word_embedding, relation_embedding)
    print graph.evaluate(test_data_producer, model_path)
'''

if __name__ == "__main__":
    tf.app.run()
