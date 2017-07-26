import tensorflow as tf
import reader

from models import gru

word_embedding_path = '../../../data/glove/glove.6B.300d.txt'
relation_embedding_path = '../../data/transE/relation_embeddings.txt'
train_path = "../../data/relation/relation_train.txt"
valid_path = "../../data/relation/relation_valid.txt"
test_path = "../../data/relation/relation_test.txt"

def main(_):
    word_to_id, word_embedding = reader.load_vocabulary(word_embedding_path)
    relation_embedding = reader.load_relation_embeddings(relation_embedding_path)

    train_data_producer = reader.DataProducer(word_to_id, train_path, 1024)
    valid_data_producer = reader.DataProducer(word_to_id, valid_path, 1024, False)

    graph = gru.GRU(len(word_embedding), len(relation_embedding),
                    word_embedding, relation_embedding)
    graph.train(train_data_producer, valid_data_producer, 100)

if __name__ == "__main__":
    tf.app.run()
