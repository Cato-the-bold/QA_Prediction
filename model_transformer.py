import sys
import tensorflow as tf
import numpy as np

import datas

VOCABULARY_SIZE = 20000
EMBEDDING_SIZE = 300

K = 32

from official.transformer.model import model_params, model_utils, transformer
def transformer_encoder(input, lengths):
    # Set up estimator and params
    params = model_params.BASE_PARAMS
    params["default_batch_size"] = K
    params["max_length"] = 500
    params["vocab_size"] = VOCABULARY_SIZE+1
    params["filter_size"] = 256
    params["num_hidden_layers"] = 2
    params["num_heads"] = 2
    params["hidden_size"] = EMBEDDING_SIZE

    model = transformer.Transformer(params, tf.estimator.ModeKeys.TRAIN)
    initializer = tf.variance_scaling_initializer(
        model.params["initializer_gain"], mode="fan_avg", distribution="uniform")
    with tf.variable_scope("Transformer", initializer=initializer):
        # Calculate attention bias for encoder self-attention and decoder
        # multi-headed attention layers.
        attention_bias = model_utils.get_padding_bias(inputs)
        # Run the inputs through the encoder layer to map the symbol
        # representations to continuous representations.
        encoder = model.encode(inputs, attention_bias)
        return tf.reduce_mean(encoder, 1)

    # embeddings = None
    # with tf.variable_scope("word2vec",reuse=True):
    #     embeddings = tf.get_variable("embeddings")
    # input = tf.nn.embedding_lookup(embeddings, input)
    # input = tf.expand_dims(input,2)
    # transformer = tensor2tensor.models.transformer.Transformer(None, None)
    # return transformer.encode(inputs, 1, None, None, None)

#shape:[batch, n_words]
inputs = tf.placeholder(tf.int32, shape=(None, None), name="inputs")
responses = tf.placeholder(tf.int32, shape=(None, None), name="responses")
inputs_lens = tf.placeholder(tf.int32, shape=(None), name="inputs_lens")
responses_lens = tf.placeholder(tf.int32, shape=(None), name="responses_lens")

def model_predication_op(encoder):
    with tf.variable_scope('encoder1'):
        input_embeddings = transformer_encoder(inputs, inputs_lens)

    with tf.variable_scope('encoder2'):
        responses_embeddings = transformer_encoder(responses, responses_lens)
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


def train(n_epochs, config):
    encoder = "DAN"  # "DAN" or "transformer"

    questions, answers, word2index = datas.load_dataset(dir="../datasets/amazonQA", num_words=VOCABULARY_SIZE)
    size = len(questions)

    fractions = [0.8, 0.2, 0.0]
    # fractions = [0.1, 0.02, 0.1]
    l1 = int(fractions[0] * size)
    l2 = int((fractions[0] + fractions[1]) * size)
    train_questions, test_questions, validate_questions = questions[:l1], questions[l1:l2], questions[l2:]
    train_answers, test_answers, validate_answers = answers[:l1], answers[l1:l2], answers[l2:]

    pred = model_predication_op(encoder)
    loss, training_op = model_train_op(pred)
    testing_op = pair_accuracy_op(pred)
    testing_op_top5 = pair_top_k_accuracy_op(pred, 5)
    testing_op_top10 = pair_top_k_accuracy_op(pred, 10)

    saver = tf.train.Saver()


    with tf.Session(config=config) as sess:
            tf.global_variables_initializer().run()
            # train_writer = tf.summary.FileWriter('./logs/train', sess.graph)

            for epoch in xrange(n_epochs):
                for i in xrange(len(train_questions)//K):
                    questions_batch, answers_batch = train_questions[i*K:(i+1)*K], train_answers[i*K:(i+1)*K]
                    sess.run(training_op, feed_dict={inputs: questions_batch, responses: answers_batch})

            accus_test, accus_test1, accus_test2 = [],[],[]
            for i in xrange(len(test_questions)//K):
                questions_batch, answers_batch = test_questions[i*K:(i+1)*K], test_answers[i*K:(i+1)*K]
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

            # save_path = saver.save(sess, "./qa_"+encoder+".ckpt")

if __name__ == "__main__":
    seed = 2018
    tf.set_random_seed(seed)
    np.random.seed(seed)

    tf.logging.set_verbosity(tf.logging.INFO)

    config = tf.ConfigProto()
    #GPU
    config.gpu_options.per_process_gpu_memory_fraction = 0.6
    config.gpu_options.allow_growth = True
    config.gpu_options.allocator_type = 'BFC'
    #CPU
    config.intra_op_parallelism_threads = 8
    config.inter_op_parallelism_threads = 8

    train(1,config)