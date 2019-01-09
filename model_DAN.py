import re
import zipfile
import os
import cPickle

import tensorflow as tf

import numpy as np
from functools import partial

import datas

PRE_TRAINED = True
VOCABULARY_SIZE = 10000
GLOVE_SIZE = 300
EMBEDDING_SIZE = GLOVE_SIZE

K = 32

#load GLOVE word embedding
def load_word_embedding(path_to_glove, word2index):
    count_all_words = 0

    embedding_matrix = np.zeros((VOCABULARY_SIZE+1, GLOVE_SIZE))

    with zipfile.ZipFile(path_to_glove) as z:
        with z.open("glove.840B.300d.txt") as f:
            for line in f:
                vals = line.split()
                word = vals[0]
                if word in word2index and word2index[word]<=VOCABULARY_SIZE:
                    count_all_words += 1
                    coefs = np.asarray(vals[1:], dtype='float32')
                    # coefs /= np.linalg.norm(coefs)
                    embedding_matrix[word2index[word], :] = coefs
                if count_all_words == len(word2index):
                    break

    return embedding_matrix

def DAN_dropout_layer(input, lengths, keep_prob):
    zeros = tf.zeros(tf.shape(input)[0:2], dtype=tf.int32)
    mask = tf.sequence_mask(lengths, tf.shape(input)[1], dtype=tf.int32)

    if keep_prob<1.0:
        keep_prob = tf.convert_to_tensor(keep_prob, shape = [1], dtype=tf.float32)
        random_tensor = tf.random_uniform(tf.shape(input)[0:2], dtype=tf.float32)
        mask = tf.where(tf.less(random_tensor, keep_prob), mask, zeros)

    # mask.set_shape((None, max_len)); dropout = tf.boolean_mask(input, mask) #doesn't work
    dropout = tf.multiply(input, tf.cast(tf.expand_dims(mask, 2), tf.float32))
    return tf.div(tf.reduce_mean(dropout,1), tf.sqrt(tf.cast(tf.reduce_mean(mask, 1, keepdims=True), tf.float32)))

def DAN_encoder(input, dropout=False):
    my_dense_layer = partial(
            tf.layers.dense, activation=tf.nn.relu,
            kernel_regularizer=tf.contrib.layers.l1_regularizer(0.001))

    for i in xrange(3):
        input = my_dense_layer(input, 500)
        input = tf.layers.batch_normalization(input, training=True, momentum=0.9)
        input = tf.nn.elu(input)

    dense3 = my_dense_layer(input, EMBEDDING_SIZE)
    bn3 = tf.layers.batch_normalization(dense3, training=True, momentum=0.9)
    encoder = tf.nn.elu(bn3)
    return encoder

# inputs = tf.placeholder(tf.int32, shape=(None, None), name="inputs")
# responses = tf.placeholder(tf.int32, shape=(None, None), name="responses")
inputs = tf.placeholder(tf.float32, shape=(None, EMBEDDING_SIZE), name="inputs")
responses = tf.placeholder(tf.float32, shape=(None, EMBEDDING_SIZE), name="responses")
inputs_lens = tf.placeholder(tf.int32, shape=(None), name="inputs_lens")
responses_lens = tf.placeholder(tf.int32, shape=(None), name="responses_lens")

def model_predication_op(encoder):
    with tf.name_scope('encoder1'):
        input_embeddings = DAN_encoder(inputs, inputs_lens)

    with tf.name_scope('encoder2'):
        responses_embeddings = DAN_encoder(responses, responses_lens)
        responses_embeddings2 = tf.layers.dense(responses_embeddings, EMBEDDING_SIZE, name="FC_responses")

    matrix = tf.matmul(input_embeddings, responses_embeddings2,transpose_b=True)
    return tf.nn.softmax(logits=matrix)

def model_train_op(predication):
    diag_loss = tf.diag_part(predication)
    loss = tf.reduce_mean(diag_loss, name="paired_loss")  # not shown
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    if reg_losses:
        loss = tf.add_n([loss] + reg_losses, name="loss")

    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    training_op = optimizer.minimize(loss)

    return loss, training_op

def pair_accuracy_op(predications):
    correct = tf.equal(tf.argmax(predications, 1, output_type=tf.int32), tf.range(tf.shape(predications)[0]))
    return tf.reduce_mean(tf.cast(correct, 'float'))

def pair_top_k_accuracy_op(predications, k):
    correct = tf.nn.in_top_k(predications, tf.range(tf.shape(predications)[0]), k)
    return tf.reduce_mean(tf.cast(correct, 'float'))

def train(n_epochs,config):
    encoder = "DAN" #"DAN" or "transformer"
    PRE_SUM = True

    questions, answers, word2index  = datas.load_dataset(dir=".",num_words=VOCABULARY_SIZE)
    size = len(questions)

    if PRE_TRAINED:
        EMBED_CACHE = "embed.cache"
        if not os.path.exists(EMBED_CACHE):
            embedding_file = '/home/cato/Projects/a-DNN-models/datasets/glove.840B.300d.zip'
            embed = load_word_embedding(embedding_file, word2index)

            ouf = open(EMBED_CACHE, 'w')
            cPickle.dump(embed, ouf)
            ouf.close()
        else:
            inf = open(EMBED_CACHE,"r")
            embed= cPickle.load(inf)
            inf.close()

        with tf.variable_scope("word2vec"):
            embeddings = tf.get_variable("embeddings", shape=[VOCABULARY_SIZE + 1, EMBEDDING_SIZE], dtype=tf.float32, initializer=tf.constant_initializer(embed))
    else:
        with tf.variable_scope("word2vec"):
            embeddings = tf.get_variable("embeddings", shape=[VOCABULARY_SIZE + 1, EMBEDDING_SIZE],
                                     initializer=tf.initializers.random_uniform(-1, 1), dtype=tf.float32)

    if PRE_SUM:
        for i in xrange(size):
            questions[i] = sum(embed[token] for token in questions[i])/len(questions[i])
            answers[i] = sum(embed[token] for token in answers[i])/len(answers[i])

    fractions = [0.8, 0.2, 0.0]
    # fractions = [0.1, 0.02, 0.1]
    l1 = int(fractions[0] * size)
    l2 = int((fractions[0]+fractions[1]) * size)
    train_questions,  test_questions, validate_questions= questions[:l1], questions[l1:l2], questions[l2:]
    train_answers, test_answers, validate_answers= answers[:l1], answers[l1:l2], answers[l2:]

    pred = model_predication_op(encoder)
    loss, training_op = model_train_op(pred)
    testing_op = pair_accuracy_op(pred)
    testing_op_top5 = pair_top_k_accuracy_op(pred, 2)
    testing_op_top10 = pair_top_k_accuracy_op(pred, 3)

    saver = tf.train.Saver()

    with tf.Session(config=config) as sess:
        tf.global_variables_initializer().run()
        train_writer = tf.summary.FileWriter('./logs/train ', sess.graph)

        for epoch in xrange(n_epochs):
            for i in xrange(len(train_questions)//K):
                questions_batch, answers_batch = train_questions[i*K:(i+1)*K], train_answers[i*K:(i+1)*K]
                sess.run(training_op, feed_dict={inputs: questions_batch, responses: answers_batch})

        K1 = 100
        accus_test, accus_test1, accus_test2 = [],[],[]
        for i in xrange(len(test_questions)//K1):
            questions_batch, answers_batch = test_questions[i*K1:(i+1)*K1], test_answers[i*K1:(i+1)*K1]
            acc_test,acc_test1,acc_test2 = sess.run([testing_op,testing_op_top5,testing_op_top10],feed_dict={inputs: questions_batch, responses: answers_batch})
            accus_test.append(acc_test)
            accus_test1.append(acc_test1)
            accus_test2.append(acc_test2)
            # print(i, "Test accuracy:", acc_test)
        print("Test top-1 accuracy:", sum(accus_test)/len(accus_test))
        print("Test top-5 accuracy:", sum(accus_test1)/len(accus_test1))
        print("Test top-10 accuracy:", sum(accus_test2)/len(accus_test2))
            # merge = tf.summary.merge_all()
            # train_writer.add_summary(sess.run(merge),epoch)

        save_path = saver.save(sess, "./qa_"+encoder, global_step=1)

if __name__ == "__main__":


    seed = 2018
    tf.set_random_seed(seed)
    np.random.seed(seed)

    tf.logging.set_verbosity(tf.logging.INFO)

    config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 0.6
    # config.gpu_options.allow_growth = True
    # config.gpu_options.allocator_type = 'BFC'
    config.intra_op_parallelism_threads = 8
    config.inter_op_parallelism_threads = 8

    train(3,config)
