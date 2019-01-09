import sqlite3
import pandas as pd

from dl_text import dl
from NLTK.corpus import wordnet
from NLTK import word_tokenize, pos_tag
from NLTK.stem import WordNetLemmatizer
import re

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

def varify_token(token):
    return re.match(r"^[a-zA-Z]+$", token) and (len(token) - 1)

lemmatizer = WordNetLemmatizer()
def lemmatize_sentence(sentence):
    if not sentence:
        return None
    sentence = dl.clean(sentence.lower())
    res = []
    for word, pos in pos_tag(word_tokenize(sentence)):
        wordnet_pos = get_wordnet_pos(pos) or wordnet.NOUN
        res.append(lemmatizer.lemmatize(word, pos=wordnet_pos))

    res = list(filter(varify_token, res))
    return res

def process_data():
    wordnet.ensure_loaded()

    conn = sqlite3.connect(r"F:\Transformer\data_process\database.sqlite", check_same_thread=False)
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
    all_results = cur.fetchall()
    table_name = all_results[0][0]

    def df_iter_func(conn ,table_name ,file_idx):
        spark = SparkSession.builder.config("spark.local.dir", r"E:\SparkTemp") \
            .config("spark.executor.memory", "800m").config("spark.driver.memory", "800m").master("local[1]").appName("data_preprocess_{}".format(file_idx)).getOrCreate()
        chunksize = int(2e1)
        df_iter = pd.read_sql_query("SELECT name, parent_id, body FROM {} where created_utc % 7 = {}".format(table_name, file_idx), conn, chunksize=chunksize)

        neg_num = 9
        all_req_list = []
        write_num = 0
        for idx, df in enumerate(df_iter):
            main_parent_id = df["parent_id"][0]
            req_list = []
            for i, r in df.iterrows():
                if not i:
                    req_list.append([r["name"], main_parent_id, 1, lemmatize_sentence(r["body"])])
                elif r["parent_id"] != main_parent_id:
                    req_list.append([r["name"], main_parent_id, 0, lemmatize_sentence(r["body"])])
                if len(req_list) == neg_num + 1:
                    all_req_list.extend(req_list)
                    break

            if len(all_req_list) > chunksize * 1000:
                o_df = pd.DataFrame(all_req_list, columns=["name", "parent_id", "label", "body_tokens"])
                all_req_list = []
                if not write_num:
                    spark.createDataFrame(o_df).write.mode("OverWrite").parquet("name_parent_label_body_n_{}.parquet".format(file_idx))
                else:
                    spark.createDataFrame(o_df).write.mode("append").parquet("name_parent_label_body_n_{}.parquet".format(file_idx))
                write_num += 1
                if write_num % 100 == 0:
                    print("write {} num :{}".format(file_idx ,write_num))

    import threading
    threads = []
    for idx in range(7):
        threads.append(threading.Thread(target=df_iter_func, args=(conn, table_name, idx), name="thread_{}".format(idx),
                                        ))
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    print("write all end")

def index_name_body_df(body_tokens_min_size = 5, show_detail = False):
    spark = SparkSession.builder.config("spark.local.dir", r"E:\SparkTemp") \
        .config("spark.executor.memory", "12g").config("spark.driver.memory", "12g").master("local[7]").appName("data_preprocess").getOrCreate()

    from functools import reduce
    df = reduce(lambda x, y: x.union(y), map(lambda idx: spark.read.parquet("name_parent_label_body_n_{}.parquet".format(idx)), range(6))).dropna()
    df = df.filter(size("body_tokens") > body_tokens_min_size).repartition(1000)
    df.persist()
    name_df = df.select("name").withColumnRenamed("name", "n_name")
    sample_df = df.join(name_df, how = "inner", on = df.parent_id == name_df.n_name).drop("n_name").select("name", "parent_id", "label")
    name_body_tokens_df = df.select("name", "body_tokens")

    def token_count(r):
        from collections import Counter
        return list(Counter(r["body_tokens"]).items())

    all_token_list = name_body_tokens_df.select("body_tokens").rdd.flatMap(
        token_count
    )\
        .reduceByKey(
        lambda x, y : x + y
    )

    from collections import Counter
    all_token_list = list(map(lambda t: t[0] ,Counter(dict(all_token_list.collect())).most_common(int(1e5))))
    word2idx = dict((token, idx) for idx, token in enumerate(all_token_list))

    import pickle
    with open("reddit_word2idx_n.pkl", "wb") as f:
        pickle.dump(word2idx, f)

    name_body_df = name_body_tokens_df.withColumn("body", udf(lambda x: list(map(lambda xx: word2idx[xx] ,filter(lambda w: 0 if word2idx.get(w) is None else 1, x))), ArrayType(IntegerType()))("body_tokens")).drop("body_tokens"). \
        withColumnRenamed("name", "k")
    sample_df = sample_df.join(name_body_df, how = "inner", on = sample_df.parent_id == name_body_df.k).drop("k") \
        .withColumnRenamed("body", "parent_body")
    sample_df = sample_df.join(name_body_df, how = "inner", on = sample_df.name == name_body_df.k).drop("k") \
        .withColumnRenamed("body", "name_body")
    sample_df = sample_df.select("parent_id" ,"parent_body", "name_body", "label")

    parent_id_df = sample_df.select("parent_id").distinct()
    train_parent_id_df, test_parent_id_df = parent_id_df.randomSplit([0.9, 0.1], seed=0.0)
    parent_id_df = train_parent_id_df.withColumn("type", lit("train")).union(test_parent_id_df.withColumn("type", lit("test")))
    sample_df = sample_df.join(parent_id_df, on = "parent_id", how = "inner")

    sample_df.select("parent_id" ,"parent_body", "name_body", "label", "type").write.mode("OverWrite").parquet("samples.parquet")

    if show_detail:
        print("samples count :")
        sample_df.groupby("label").agg(count("*").alias("labels_count")).show()
        sample_df.show(5)

def process_reddit_data():
    process_data()
    index_name_body_df()

def process_snli_data():
    wordnet.ensure_loaded()
    def pos_tag(sentence_parse):
        res = []
        for pos, word in re.findall(r"([a-zA-Z]+) ([a-zA-Z]+)", sentence_parse):
            wordnet_pos = get_wordnet_pos(pos) or wordnet.NOUN
            res.append(lemmatizer.lemmatize(word.lower(), pos=wordnet_pos))
        return res

    import json
    import re
    import pickle
    with open("reddit_word2idx_n.pkl", "rb") as f:
        reddit_word2idx = pickle.load(f)

    class ParseJson(object):
        def __init__(self):
            self.result = None

        def process_json(self ,name = "train"):
            from collections import Counter
            cnt = Counter()
            with open("snli_{}.txt".format(name), "w") as o:
                with open(r"F:\Transformer\data_process\snli_1.0\snli_1.0_{}.jsonl".format(name), "r") as f:
                    write_num = 0
                    while True:
                        line = f.readline().strip()
                        if not line:
                            break

                        json_obj = json.loads(line)
                        gold_label = json_obj["gold_label"]
                        sentence1_parse = pos_tag(json_obj["sentence1_parse"])
                        sentence2_parse = pos_tag(json_obj["sentence2_parse"])
                        cnt.update(sentence1_parse + sentence2_parse)

                        o.write("{}\t{}\t{}\n".format(" ".join(sentence1_parse), " ".join(sentence2_parse), gold_label))
                        write_num += 1
                        if write_num % 100000 == 0:
                            print("{} write {} end".format(name, write_num))
            self.result = cnt

        def idx_file(self, name = "train", word2idx =None):
            write_num = 0
            label_dict = {
                "contradiction": 0,
                "neutral": 1,
                "entailment": 2
            }

            with open("snli_{}_idx.txt".format(name), "w") as o:
                with open("snli_{}.txt".format(name), "r") as f:
                    while True:
                        line = f.readline().strip()
                        if not line:
                            break
                        sent1, sent2, label = line.split("\t")
                        label = label_dict.get(label)
                        if label is None:
                            continue
                        sent1_str = " ".join(map(lambda xx: str(word2idx[xx]) ,filter(lambda w: 0 if word2idx.get(w) is None else 1, sent1.split(" "))))
                        sent2_str = " ".join(map(lambda xx: str(word2idx[xx]) ,filter(lambda w: 0 if word2idx.get(w) is None else 1, sent2.split(" "))))
                        if not sent1_str or not sent2_str:
                            continue
                        o.write("{}\t{}\t{}\n".format(sent1_str, sent2_str, label))
                        write_num += 1
                        if write_num % 100000 == 0:
                            print("{} write {} end".format(name, write_num))

        def get_result(self):
            return self.result

    import threading
    threads = []
    for name in ["train", "test"]:
        exec("pj_{} = ParseJson()".format(name))
        exec('threads.append(threading.Thread(target=pj_{}.process_json, args=(\"{}\",), name="thread_{}",\
                                        ))'.format(name, name, name))
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    print("data_process end")

    cnt_list = []
    for name in ["train", "test"]:
        exec("cnt_list.append(pj_{}.result)".format(name))

    from collections import defaultdict, Counter
    cnt_items = defaultdict(int)
    for cnt in cnt_list:
        for k, v in cnt.items():
            cnt_items[k] += v
    snli_req_tokens = list(map(lambda t: t[0] ,Counter(dict(cnt_items)).most_common(int(1e5))))

    print("req_tokens :")
    print(len(snli_req_tokens))

    snli_add_tokens = set(snli_req_tokens).difference(set(list(reddit_word2idx.keys())))

    print("add tokens :")
    print(len(snli_add_tokens))
    print("the len of snli_add_tokens indicate new informations")

    word2idx_items = list(reddit_word2idx.items()) + [(word, idx + len(reddit_word2idx)) for idx, word in enumerate(snli_add_tokens)]
    word2idx = dict(word2idx_items)
    with open("word2idx_n.pkl", "wb") as f:
        pickle.dump(word2idx, f)

    threads = []
    for name in ["train", "test"]:
        threads.append(threading.Thread(target=ParseJson.idx_file, args=(name, word2idx), name="thread_{}".format(name),
                                        ))
    threads = []
    for name in ["train", "test"]:
        exec('threads.append(threading.Thread(target=pj_{}.idx_file, args=(\"{}\", word2idx), name="idx_thread_{}",\
                                        ))'.format(name, name, name))

    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    print("data_idx end")

if __name__ == "__main__":
    process_reddit_data()
    process_snli_data()


import tensorflow as tf
import numpy as np
from tensor2tensor.layers import common_attention

class Transformer(object):
    def __init__(self, input, word_size = 500000, embedding_dim = 30, batch_num = 10, use_position_encoding = False,
                 use_diy = True):
        self.input = input
        self.word_size = word_size
        self.embedding_dim = embedding_dim
        self.batch_num = batch_num

        with tf.name_scope("word_embedding"):
            self.word_W = tf.get_variable(name="word_W",
                shape=[word_size, embedding_dim],
                initializer=tf.orthogonal_initializer()
            )

        self.embedding_input = tf.nn.embedding_lookup(self.word_W, self.input)

        if use_position_encoding:
            if use_diy:
                self.postional_encoding_add = self.positional_encoding_layer(input=self.embedding_input)
            else:
                self.postional_encoding_add = common_attention.add_timing_signal_1d(self.embedding_input)
        else:
            self.postional_encoding_add = self.embedding_input

        self.output = tf.reduce_mean(self.multi_encoding_layer(self.postional_encoding_add), axis=-1)

    # input of encoding [batch, seq_len, embed_dim]
    def positional_encoding_layer(self, input):
        # construct positional encoding layer by numpy
        first_dim = int(input.get_shape()[0])
        embed_dim = int(input.get_shape()[-1])
        seq_len = int(input.get_shape()[-2])

        req = np.zeros(shape=[seq_len, embed_dim])
        for pos in range(req.shape[0]):
            req[pos, :] = pos
            for j in range(len(req[pos, :])):
                req[pos, j] /= np.power(10000, int(j/2) / embed_dim)
                if int(j/2) == j/2:
                    req[pos, j] = np.sin(req[pos, j])
                else:
                    req[pos, j] = np.cos(req[pos, j])

        req = [req.tolist()] * first_dim

        postional_encoding = tf.convert_to_tensor(req, dtype=tf.float32)
        assert postional_encoding.get_shape() == input.get_shape()

        return input + tf.Variable(postional_encoding, trainable=False)

    def scaled_dot_product_attension_layer(self, Q_batch, K_batch, V_batch):
        def sample_attension(Q, K, V):
            dk = int(Q.get_shape()[-1])
            return tf.matmul(tf.nn.softmax(tf.matmul(Q, tf.transpose(K, [1, 0])) / dk, dim=-1), V)

        req_list = []
        for batch_idx in range(self.batch_num):
            Q, K, V = map(lambda M: tf.squeeze(tf.slice(M ,[batch_idx, 0, 0], [1, -1, -1])), [Q_batch, K_batch, V_batch])
            req_list.append(tf.expand_dims(sample_attension(Q, K, V), 0))

        return tf.concat(req_list, axis=0)

    def multihead_attension_layer(self, Q_batch, K_batch, V_batch, h = 3, dk = 10, dv = 10, d_model = 10 * 3):
        assert d_model == int(Q_batch.get_shape()[-1]) == int(K_batch.get_shape()[-1]) == int(V_batch.get_shape()[-1])
        d_model = dk * h

        def map_to_attension_format(qkv, W):
            last_dim = int(qkv.get_shape()[-1])
            seq_len = int(qkv.get_shape()[-2])
            w_dim = int(W.get_shape()[-1])
            return tf.reshape(tf.matmul(tf.reshape(qkv, [-1, last_dim]), W), [-1, seq_len, w_dim])

        head_list = []
        for head_idx in range(h):
            with tf.variable_scope("multihead_attension_{}".format(head_idx), reuse=tf.AUTO_REUSE):
                WQ = tf.get_variable(shape=[d_model, dk], name="WQ", initializer=tf.orthogonal_initializer())
                WK = tf.get_variable(shape=[d_model, dk], name="WK", initializer=tf.orthogonal_initializer())
                WV = tf.get_variable(shape=[d_model, dv], name="WV", initializer=tf.orthogonal_initializer())

                Q = map_to_attension_format(Q_batch, WQ)
                K = map_to_attension_format(K_batch, WK)
                V = map_to_attension_format(V_batch, WV)

                head = self.scaled_dot_product_attension_layer(Q, K, V)
                head_list.append(head)
        head_concat = tf.concat(head_list, axis=-1)
        head_concat_dim = int(head_concat.get_shape()[-1])
        seq_len = int(head_concat.get_shape()[-2])
        with tf.variable_scope("WO_layer", reuse=tf.AUTO_REUSE):
            WO = tf.get_variable(shape=[head_concat_dim, d_model], name="WO", initializer=tf.orthogonal_initializer())

        return tf.reshape(tf.matmul(tf.reshape(head_concat, [-1, head_concat_dim]), WO), [-1, seq_len, d_model])

    def add_norm(self, x, sub_layer):
        assert x.get_shape() == sub_layer.get_shape()
        return tf.contrib.layers.layer_norm(x + sub_layer)

    def FFN(self, input, w1_dim = 100):
        d_model = int(input.get_shape()[-1])
        with tf.variable_scope("FFN_layer", reuse=tf.AUTO_REUSE):
            W1 = tf.get_variable(shape=[d_model, w1_dim], name="W1", initializer=tf.orthogonal_initializer())
            b1 = tf.get_variable(shape=[w1_dim], name="b1", initializer=tf.constant_initializer(1.0))
            W2 = tf.get_variable(shape=[w1_dim, d_model], name="W2", initializer=tf.orthogonal_initializer())
            b2 = tf.get_variable(shape=[d_model], name="b2", initializer=tf.constant_initializer(1.0))
        seq_len = int(input.get_shape()[-2])
        return tf.reshape(tf.nn.xw_plus_b(tf.nn.relu(tf.nn.xw_plus_b(tf.reshape(input, [-1, d_model]), W1, b1)), W2, b2), [-1, seq_len, d_model])

    def multi_encoding_layer(self, input, layer_num = 6):
        input_list = [input] * 3
        ffn_add_norm = None
        for num in range(layer_num):
            with tf.name_scope("single_encoding_layer_{}".format(num)):
                with tf.name_scope("add_norm_1"):
                    multihead_attension = self.multihead_attension_layer(input_list[0], input_list[1], input_list[2])
                    with tf.variable_scope("norm_1{}".format(num)):
                        multihead_output = self.add_norm(input_list[0] ,multihead_attension)
                with tf.name_scope("add_norm_2"):
                    ffn_output = self.FFN(multihead_output)
                    with tf.variable_scope("norm_2{}".format(num)):
                        ffn_add_norm = self.add_norm(multihead_output ,ffn_output)
                input_list = [ffn_add_norm] * 3
        return ffn_add_norm