import json
import re
import os
import sys
import random
import gzip
import zipfile
import math
import string
import collections
import logging
import itertools
import cPickle
from functools import partial

import numpy as np
import tensorflow as tf

import datas
import TFgenerator

VOCABULARY_SIZE = 10000
GLOVE_SIZE = 300
EMBEDDING_SIZE = GLOVE_SIZE

K = 32

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

def DAN_dropout_layer(input, lengths, keep_prob):
    zeros = tf.zeros(tf.shape(input)[0:2], dtype=tf.int32)
    mask = tf.sequence_mask(lengths, tf.shape(input)[1], dtype=tf.int32)

    if keep_prob<1.0:
        keep_prob = tf.convert_to_tensor(keep_prob, shape = [1], dtype=tf.float64)
        random_tensor = tf.random_uniform(tf.shape(input)[0:2], dtype=tf.float64)
        mask = tf.where(tf.less(random_tensor, keep_prob), mask, zeros)

    # mask.set_shape((None, max_len)); dropout = tf.boolean_mask(input, mask) #doesn't work
    dropout = tf.multiply(input, tf.cast(tf.expand_dims(mask, 2), tf.float64))
    return tf.div(tf.reduce_mean(dropout,1), tf.sqrt(tf.cast(tf.reduce_mean(mask, 1, keepdims=True), tf.float64)))

def DAN_encoder_v2(input, lengths, dropout=False):
    with tf.variable_scope("word2vec",reuse=tf.AUTO_REUSE):
        embeddings = tf.get_variable("embeddings", shape=[VOCABULARY_SIZE + 1, EMBEDDING_SIZE],
                                     initializer=tf.initializers.random_uniform(-0.25, 0.25), dtype=tf.float32,
                                     trainable=True)
    input = tf.nn.embedding_lookup(embeddings, input)
    input = DAN_dropout_layer(input,lengths,keep_prob=1.0)
    input.set_shape((None,EMBEDDING_SIZE))

    my_dense_layer = partial(
            tf.layers.dense, activation=tf.nn.relu,
            kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001))

    for i in xrange(3):
        input = my_dense_layer(input, 500)
        input = tf.layers.batch_normalization(input, training=True, momentum=0.9)
        input = tf.nn.elu(input)

    dense3 = my_dense_layer(input, EMBEDDING_SIZE)
    bn3 = tf.layers.batch_normalization(dense3, training=True, momentum=0.9)
    encoder = tf.nn.elu(bn3)
    return encoder

inputs = tf.placeholder(tf.int32, shape=(None, None), name="inputs")
responses = tf.placeholder(tf.int32, shape=(None, None), name="responses")
inputs_lens = tf.placeholder(tf.int32, shape=(None), name="inputs_lens")
responses_lens = tf.placeholder(tf.int32, shape=(None), name="responses_lens")

# def cnn_model_fn(features, labels, mode):
def qa_model_fn(features, labels, mode, params):
    with tf.name_scope('encoder1'):
        input_embeddings = DAN_encoder_v2(features['question'], features['q_len'])

    with tf.name_scope('encoder2'):
        responses_embeddings = DAN_encoder_v2(features['answer'], labels)
        responses_embeddings2 = tf.layers.dense(responses_embeddings, EMBEDDING_SIZE, name="FC_responses")

    matrix = tf.matmul(input_embeddings, responses_embeddings2,transpose_b=True)
    predications = tf.nn.softmax(logits=matrix, name="softmax_tensor")

    diag_loss = tf.diag_part(predications)

    loss = tf.reduce_mean(diag_loss, name="paired_loss")
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    if reg_losses:
        loss = tf.add_n([loss] + reg_losses, name="loss")

    if mode == tf.estimator.ModeKeys.TRAIN:
        # global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    elif mode == tf.estimator.ModeKeys.EVAL:
        eval_metric_ops = {
            "accuracy": tf.metrics.accuracy(
                labels=tf.range(tf.shape(predications)[0]), predictions=tf.argmax(predications, 1, output_type=tf.int32)),
            "top-5 accuracy": tf.metrics.accuracy(
                labels=tf.fill((tf.shape(predications)[0]), True),
                predictions=tf.nn.in_top_k(predications, tf.range(tf.shape(predications)[0]), 5))
        }
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
    elif mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions={"probabilities":predications})

def train(n_epochs,config):

    train_tfrecord = "train.tfrecord"
    test_tfrecord = "test.tfrecord"

    if not os.path.exists(train_tfrecord):
        questions, answers, word2index = datas.load_dataset(dir=".",num_words=VOCABULARY_SIZE)
        size = len(questions)

        fractions = [0.8, 0.2, 0.0]
        # fractions = [0.1, 0.02, 0.1]
        l1 = int(fractions[0] * size)
        l2 = int((fractions[0] + fractions[1]) * size)

        TFgenerator.write_tfrecord(train_tfrecord, questions[:l1], answers[:l1])
        TFgenerator.write_tfrecord(test_tfrecord, questions[l1:l2], answers[l1:l2])

    train_iter = TFgenerator.load_datasets(train_tfrecord, n_epochs, K)
    test_iter = TFgenerator.load_datasets(test_tfrecord, 1, K)

    checkpointing_config = tf.estimator.RunConfig(
        save_checkpoints_secs=20 * 60,  # Save checkpoints every 20 minutes.
        keep_checkpoint_max=10,  # Retain the 10 most recent checkpoints.
        save_checkpoints_steps=None,
        save_summary_steps=5
    )

    dan_classifier = tf.estimator.Estimator(
            model_fn=qa_model_fn, model_dir="/tmp/DAN_model", params={}, config=checkpointing_config)

    tensors_to_log = {"predictions": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=100)
    from tensorflow.python import debug as tf_debug
    hooks = [tensors_to_log, tf_debug.LocalCLIDebugHook()]

    dan_classifier.train(input_fn=lambda: train_iter, hooks=hooks)

    print dan_classifier.evaluate(input_fn=lambda: test_iter, hooks=hooks)

    def serving_input_receiver_fn():
        features = {
            'question': tf.VarLenFeature(tf.int64),
            'answer': tf.VarLenFeature(tf.int64),
            'q_len': tf.FixedLenFeature([], tf.int64),
            'a_len': tf.FixedLenFeature([], tf.int64)
        }
        serialized_tf_example = tf.placeholder(dtype=tf.string,
                                             name='input_example_tensor')
        receiver_tensors = {'examples': serialized_tf_example}
        features = tf.parse_example(serialized_tf_example, features)
        return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)

    dan_classifier.export_savedmodel(".", serving_input_receiver_fn)

def main(argv):
    config = tf.ConfigProto()
    #GPU
    config.gpu_options.per_process_gpu_memory_fraction = 0.6
    config.gpu_options.allow_growth = True
    config.gpu_options.allocator_type = 'BFC'
    #CPU
    config.intra_op_parallelism_threads = 8
    config.inter_op_parallelism_threads = 8

    train(1,config)

if __name__ == '__main__':
    seed = 2018
    tf.set_random_seed(seed)
    np.random.seed(seed)

    tf.logging.set_verbosity(tf.logging.INFO)

    tf.app.run(main)