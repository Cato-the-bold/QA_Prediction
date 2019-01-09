import os
import sys
import math
import collections
import cPickle

import tensorflow as tf

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def _float_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def read_tfrecord(filenames,batch_num):
    tfrecord_file_queue = tf.train.string_input_producer(filenames, name='queue', num_epochs=batch_num)
    reader = tf.TFRecordReader()
    _, tfrecord_serialized = reader.read(tfrecord_file_queue)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    coord.request_stop()
    coord.join(threads)

def write_tfrecord(filename, questions, answers):
    writer = tf.python_io.TFRecordWriter(filename)

    for q,a in zip(questions,answers):
        example = tf.train.Example(features=tf.train.Features(feature={
                    'question': _int64_feature(q),
                    'answer': _int64_feature(a),
                    'q_len': _int64_feature([len(q)]),
                    'a_len': _int64_feature([len(a)])
                    }))
        writer.write(example.SerializeToString())
    writer.close()

def load_datasets(filenames, batch_num, batch_size):
    def _input_func(exmaple):
        features = tf.parse_single_example(exmaple,
                        features={
                            'question': tf.VarLenFeature(tf.int64),
                            'answer': tf.VarLenFeature(tf.int64),
                            'q_len': tf.FixedLenFeature([],tf.int64),
                            'a_len': tf.FixedLenFeature([],tf.int64)
                        }, name='features')
        a = features.pop('a_len')
        features['question'] = tf.sparse.to_dense(features['question'])
        features['answer'] = tf.sparse.to_dense(features['answer'])
        return features,a

    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(_input_func).repeat(batch_num)
    dataset = dataset.padded_batch(batch_size, padded_shapes=({"question":[None],"answer":[None], "q_len":[]}, []))
    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()
    return next_element

