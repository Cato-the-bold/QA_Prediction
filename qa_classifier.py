import sys
import tensorflow as tf
import numpy as np

VOCABULARY_SIZE = 20000
EMBEDDING_SIZE = 300

K = 32

#shape:[batch, n_words]
inputs = tf.placeholder(tf.int32, shape=(None, None), name="inputs")
responses = tf.placeholder(tf.int32, shape=(None, None), name="responses")
inputs_lens = tf.placeholder(tf.int32, shape=(None), name="inputs_lens")
responses_lens = tf.placeholder(tf.int32, shape=(None), name="responses_lens")

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

    with tf.Session(config=config) as sess:
            tf.global_variables_initializer().run()
            # train_writer = tf.summary.FileWriter('./logs/train', sess.graph)

            for epoch in xrange(n_epochs):
                for i in xrange(len(train_questions)//K):
                    questions_batch, answers_batch = train_questions[i*K:(i+1)*K], train_answers[i*K:(i+1)*K]
                    sess.run(training_op, feed_dict={inputs: questions_batch, responses: answers_batch})

            accus_test = []
            for i in xrange(len(test_questions)//K):
                questions_batch, answers_batch = test_questions[i*K:(i+1)*K], test_answers[i*K:(i+1)*K]
                acc_test = sess.run([testing_op],feed_dict={inputs: questions_batch, responses: answers_batch})
                accus_test.append(acc_test)
                # print(i, "Test accuracy:", acc_test)
            print("Test top-1 accuracy:", sum(accus_test)/len(accus_test))

if __name__ == "__main__":
    seed = 2018
    tf.set_random_seed(seed)
    np.random.seed(seed)

    tf.logging.set_verbosity(tf.logging.INFO)

    config = tf.ConfigProto()
    #GPU
    config.gpu_options.per_process_gpu_memory_fraction = 0.8
    config.gpu_options.allow_growth = True
    config.gpu_options.allocator_type = 'BFC'
    #CPU
    config.intra_op_parallelism_threads = 8
    config.inter_op_parallelism_threads = 8

    train(1,config)