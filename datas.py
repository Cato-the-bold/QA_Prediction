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
import fnmatch
import numpy as np
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
# import spacy
# nlp = spacy.load("en", disable=['parser', 'tagger', 'ner'])

dump_file = 'data.dump'

def parse_gzip(path):
  g = gzip.open(path, 'rb')
  for l in g:
    yield eval(l)

def get_content(path, category):
    res = []
    for row in parse_gzip(path):
        res.append([row['question'], row['answer']])
    return res

# load data from json files.
def preprocess_datasets(dir, file_filter, num_words, max_len):
    data = []
    fl = fnmatch.filter(os.listdir(dir), file_filter)
    for f in fl:
        data.extend(get_content(os.path.join(dir,f)))
    random.shuffle(data)

    tokenizer = Tokenizer(num_words)
    questions,answers = list(zip(*data))
    tokenizer.fit_on_texts(questions+answers)
    X = tokenizer.texts_to_sequences(questions)
    Y = tokenizer.texts_to_sequences(answers)

    #if X[i] or Y[i] is None, remove the pair.
    X,Y,_ = zip(*list((x,y) for x,y in zip(X,Y) if 0<len(x)<=max_len and 0<len(y)<=max_len))
    X = list(X)
    Y = list(Y)

    ouf = open(dump_file, 'wb')
    cPickle.dump(X,ouf)
    cPickle.dump(Y,ouf)
    cPickle.dump({k:v for k,v in tokenizer.word_index.items() if v<=num_words},ouf)
    ouf.close()

    #statistics:
    # print collections.Counter([len(r) for r in X])[0]
    # print collections.Counter([len(r) for r in Y])[0]

    return X, Y, tokenizer.word_index

def pad_batch(matrix):
    # print matrix
    lengths = [len(r) for r in matrix]
    max_len = max(lengths)
    for i,row in enumerate(matrix):
        matrix[i]+=[0]*(max_len-len(row))
    return matrix, lengths

def load_dataset(dir,file_filter,num_words,max_len=500):
    if not os.path.isfile(os.path.join(dump_file)):
        return preprocess_datasets(dir,file_filter,num_words,max_len)
    else:
        inf = open(dump_file)
        X = cPickle.load(inf)
        Y = cPickle.load(inf)
        word2index = cPickle.load(inf)
        inf.close()
        return X,Y,word2index

def get_sentence_batch(K, questions, answers):
    indices = np.random.choice(len(questions),K).tolist()
    return [questions[i] for i in indices], [answers[i] for i in indices]


#load GLOVE word embedding
def load_word_embedding(path_to_glove, word2index, n_vocabulary, glove_size = 300):
    count_all_words = 0

    embedding_matrix = np.zeros((n_vocabulary + 1, glove_size))

    with zipfile.ZipFile(path_to_glove) as z:
        with z.open("glove.840B.300d.txt") as f:
            for line in f:
                vals = line.split()
                word = vals[0]
                if word in word2index and word2index[word]<=n_vocabulary:
                    count_all_words += 1
                    coefs = np.asarray(vals[1:], dtype='float32')
                    # coefs /= np.linalg.norm(coefs)
                    embedding_matrix[word2index[word], :] = coefs
                if count_all_words == len(word2index):
                    break

    return embedding_matrix