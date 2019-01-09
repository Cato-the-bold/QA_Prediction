from __future__ import print_function

import os
import cPickle
import argparse

import tensorflow as tf
from tensorflow.python.saved_model import tag_constants

import numpy as np
from functools import partial

import datas

DEBUG_MODE = False
VOCABULARY_SIZE = 10000
GLOVE_SIZE = 300
EMBEDDING_SIZE = GLOVE_SIZE

K = 32

inputs = tf.placeholder(tf.int32, shape=(None, None), name="inputs")
responses = tf.placeholder(tf.int32, shape=(None, None), name="responses")
inputs_lens = tf.placeholder(tf.int32, shape=(None), name="inputs_lens")
responses_lens = tf.placeholder(tf.int32, shape=(None), name="responses_lens")

def DAN_dropout_layer(input, lengths, keep_prob):
    zeros = tf.zeros(tf.shape(input)[0:2], dtype=tf.int32)
    keep_prob_tensor = tf.convert_to_tensor(keep_prob, dtype=tf.float32)

    if keep_prob<1.0:
        random_tensor = tf.random_uniform(tf.shape(input)[0:2], dtype=tf.float32)
        input = tf.where(tf.less(random_tensor, keep_prob), input, zeros)

    return tf.div(tf.reduce_mean(input,1), tf.cast(tf.expand_dims(lengths,1), tf.float32))

def DAN_encoder_v2(input_, lengths, embeddings=None, dropout=False):
    # version 1
    # with tf.variable_scope("word2vec",reuse=True):
    #     embeddings = tf.get_variable("embeddings")

    tf.logging.info("shape of embeddings:{}".format(tf.shape(embeddings)))
    input = tf.nn.embedding_lookup(embeddings, input_)
    embed_zeros = tf.zeros_like(input, dtype=np.float32)
    input = tf.where(tf.equal(tf.tile(tf.expand_dims(input_,2), multiples=(1,1,EMBEDDING_SIZE)),tf.constant(0)), embed_zeros,input)

    input = DAN_dropout_layer(input,lengths,keep_prob=1.0)
    input.set_shape((None,EMBEDDING_SIZE))

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

def model_predication_op(embeddings):
    with tf.name_scope('encoder1'):
        input_embeddings = DAN_encoder_v2(inputs, inputs_lens, embeddings)

    with tf.name_scope('encoder2'):
        responses_embeddings = DAN_encoder_v2(responses, responses_lens, embeddings)
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
    training_op = optimizer.minimize(loss,global_step=tf.train.get_or_create_global_step())

    return loss, training_op

def pair_accuracy_op(predications):
    correct = tf.equal(tf.argmax(predications, 1, output_type=tf.int32), tf.range(tf.shape(predications)[0]))
    return tf.reduce_mean(tf.cast(correct, 'float'))

def pair_top_k_accuracy_op(predications, k):
    correct = tf.nn.in_top_k(predications, tf.range(tf.shape(predications)[0]), k)
    return tf.reduce_mean(tf.cast(correct, 'float'))

def train(n_epochs, config, predict):
    encoder = "DAN" #"DAN" or "transformer"

    questions, answers, word2index  = datas.load_dataset(dir="../datasets/amazonQA", file_filter="qa_Electronics.json.gz", num_words=VOCABULARY_SIZE)
    size = len(questions)

    # version 1
    # with tf.variable_scope("word2vec"):
    #     embeddings = tf.get_variable("embeddings", shape=[VOCABULARY_SIZE + 1, EMBEDDING_SIZE],
    #                                  initializer=tf.initializers.random_uniform(-0.25, 0.25), dtype=tf.float32,
    #                                  trainable=True)

    # version 2
    with tf.variable_scope("word2vec"):
        pretrained_embs = tf.get_variable(
            name="zeros",
            initializer=tf.constant_initializer(np.zeros([1,EMBEDDING_SIZE]), dtype=tf.float32),
            shape=[1, EMBEDDING_SIZE],
            trainable=False)
        train_embeddings = tf.get_variable(
            name="trainable",
            shape=[VOCABULARY_SIZE, EMBEDDING_SIZE],
            initializer=tf.random_uniform_initializer(-0.25, 0.25), dtype=tf.float32,
            trainable=True)

        embeddings = tf.concat([pretrained_embs, train_embeddings], axis=0)

    fractions = [0.8, 0.2, 0.0]
    # fractions = [0.1, 0.02, 0.1]
    l1 = int(fractions[0] * size)
    l2 = int((fractions[0]+fractions[1]) * size)
    train_questions,  test_questions, validate_questions= questions[:l1], questions[l1:l2], questions[l2:]
    train_answers, test_answers, validate_answers= answers[:l1], answers[l1:l2], answers[l2:]

    pred = model_predication_op(embeddings)
    loss, training_op = model_train_op(pred)
    # global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
    testing_op = pair_accuracy_op(pred)
    testing_op_top5 = pair_top_k_accuracy_op(pred, 5)
    testing_op_top10 = pair_top_k_accuracy_op(pred, 10)

    chkpoint_saver = tf.train.Saver(max_to_keep=50, )

    if not predict:
        with tf.Session(config=config) as sess:
            if DEBUG_MODE:
                from tensorflow.python import debug as tf_debug
                sess = tf_debug.LocalCLIDebugWrapperSession(sess)

                def my_filter_callable(datum, tensor):
                    # A filter that detects zero-valued scalars.
                    return len(tensor.shape) == 0 and tensor == 0.0

                sess.add_tensor_filter('my_filter', my_filter_callable)

            tf.global_variables_initializer().run()

            train_writer = tf.summary.FileWriter('./logs/train', sess.graph)

            for epoch in xrange(n_epochs):
                for i in xrange(len(train_questions)//K):
                    questions_batch, answers_batch = datas.get_sentence_batch(K, train_questions, train_answers)
                    questions_batch, q_lens = datas.pad_batch(questions_batch)
                    answers_batch, a_lens = datas.pad_batch(answers_batch)

                    sess.run(training_op, feed_dict={inputs: questions_batch, responses: answers_batch, inputs_lens: q_lens, responses_lens: a_lens})
                if (i + 1) % 50 == 0:
                    chkpoint_saver.save(sess, '.logs/train/qa_model', tf.train.get_global_step().eval())

            accus_test, accus_test1, accus_test2 = [],[],[]
            for i in xrange(len(test_questions)//K):
                questions_batch, answers_batch = datas.get_sentence_batch(K, test_questions, test_answers)
                questions_batch, q_lens = datas.pad_batch(questions_batch)
                answers_batch, a_lens = datas.pad_batch(answers_batch)

                acc_test,acc_test1,acc_test2 = sess.run([testing_op,testing_op_top5,testing_op_top10],
                                                        feed_dict={inputs: questions_batch, responses: answers_batch, inputs_lens: q_lens, responses_lens: a_lens})
                accus_test.append(acc_test)
                accus_test1.append(acc_test1)
                accus_test2.append(acc_test2)
                # print(i, "Test accuracy:", acc_test)
            tf.summary.scalar("top-1-accuracy", tf.reduce_mean(accus_test))
            tf.summary.scalar("top-5-accuracy", tf.reduce_mean(accus_test1))
            tf.summary.scalar("top-10-accuracy", tf.reduce_mean(accus_test2))

            print("[Train] accuracy:", sum(accus_test)/len(accus_test))

            merge = tf.summary.merge_all()
            train_writer.add_summary(sess.run(merge),tf.train.get_global_step().eval())
            train_writer.close()

            with tf.variable_scope("word2vec", reuse=True):
                # embeddings = tf.get_variable("embeddings")
                print(sess.run(embeddings[:5,:]))
            # exit(0)

            Inputs = {"inputs":inputs,"responses":responses,"inputs_lens":inputs_lens,"responses_lens":responses_lens}
            Outputs = {"testing_op":testing_op}
            tf.saved_model.simple_save(
                sess, 'saved_model', Inputs, Outputs
            )
# Load Model
    else:
        with tf.Session(config=config) as sess:
            if DEBUG_MODE:
                from tensorflow.python import debug as tf_debug
                sess = tf_debug.LocalCLIDebugWrapperSession(sess)

            tf.global_variables_initializer().run()

            tf.saved_model.loader.load(
                sess,
                [tag_constants.SERVING],
                'saved_model',
            )

            # graph = sess.graph
            # embeddings = graph.get_tensor_by_name('word2vec/embeddings:0')

            accus_test = []
            for i in xrange(len(test_questions)//K):
                questions_batch, answers_batch = datas.get_sentence_batch(K, test_questions, test_answers)
                questions_batch, q_lens = datas.pad_batch(questions_batch)
                answers_batch, a_lens = datas.pad_batch(answers_batch)

                acc_test = sess.run(testing_op_top5,feed_dict={inputs: questions_batch, responses: answers_batch, inputs_lens: q_lens, responses_lens: a_lens})
                accus_test.append(acc_test)
            print("[Evaluate] Top-5 accuracy:", sum(accus_test)/len(accus_test))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument(
        "--ps_hosts",
        type=str,
        default="",
        help="Comma-separated list of hostname:port pairs"
    )
    parser.add_argument(
        "--task_index",
        type=int,
        default=0,
        help="Index of task within the job"
    )
    FLAGS, unparsed = parser.parse_known_args()

    seed = 2018
    tf.set_random_seed(seed)
    np.random.seed(seed)

    tf.logging.set_verbosity(tf.logging.INFO)

    config = tf.ConfigProto(log_device_placement=False)
    config.gpu_options.per_process_gpu_memory_fraction = 0.95
    config.gpu_options.allow_growth = True
    config.gpu_options.allocator_type = 'BFC'
    config.intra_op_parallelism_threads = 8
    config.inter_op_parallelism_threads = 8

    train(5, config, predict=False)
