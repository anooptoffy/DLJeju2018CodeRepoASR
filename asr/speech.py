import os
import time
import random
import numpy as np
import pandas as pd
import librosa

import tensorflow as tf
from tensorflow.python.platform import gfile

# parameters
PATH_TRAIN = 'data/train'
PATH_TEST = 'data/test'
BATCH_SIZE = 100
ITERATIONS = 500
ITERATIONS_TEST = 10
EVAL_EVERY = 5
HEIGHT = 20
WIDTH = 44
NUM_LABELS = 0
LEARNING_RATE = 1E-4
LOGDIR = 'log/'
TEST_LOGDIR = 'log_test/'
LABEL_TO_INDEX_MAP = {}

# map from labels to index and vice versa
def init(path):
    labels = os.listdir(path)
    index = 0
    for label in labels:
        LABEL_TO_INDEX_MAP[label] = index
        index += 1

    global NUM_LABELS
    NUM_LABELS = len(LABEL_TO_INDEX_MAP)

# label to vector
def one_hot_encoding(label):
    encoding = [0] * len(LABEL_TO_INDEX_MAP)
    encoding[LABEL_TO_INDEX_MAP[label]] = 1
    return encoding

# get MFCC
def get_mfcc(wave_path, PAD_WIDTH=WIDTH):
    wave, sr = librosa.load(wave_path, mono=True)
    mfccs = librosa.feature.mfcc(y=wave, sr=sr, n_mfcc=HEIGHT)
    mfccs = np.pad(mfccs, ((0,0), (0, PAD_WIDTH - len(mfccs[0]))), mode='constant')
    return mfccs

# get next batch of labels and audio features
def get_batch(batch_size, path):
    X = []
    Y = []
    random.seed(5896)
    path = os.path.join(path, '*', '*.wav')
    waves = gfile.Glob(path)
    while True:
        random.shuffle(waves)
        for wave_path in waves:
            _, label = os.path.split(os.path.dirname(wave_path))
            X.append(get_mfcc(wave_path))
            Y.append(one_hot_encoding(label))
            if (len(X) == batch_size):
                yield X, Y
                X = []
                Y = []

def get_model(input, dropout):
    # first convolutional layer
    with tf.name_scope('Conv1'):
        input_4D = tf.reshape(input, [-1, HEIGHT, WIDTH, 1])
        w1 = tf.Variable(tf.truncated_normal([12, 8, 1, 44], stddev=0.01), name='W')
        b1 = tf.Variable(tf.zeros([44]), name='B')
        conv1 = tf.nn.conv2d(input_4D, w1, strides=[1, 1, 1, 1], padding='SAME')
        act1 = tf.nn.bias_add(conv1, b1)
        drop1 = tf.nn.dropout(act1, dropout)
        max_pool1 = tf.nn.max_pool(drop1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        tf.summary.histogram("weights", w1)
        tf.summary.histogram("biases", b1)
        tf.summary.histogram("activations", act1)
        tf.summary.histogram("dropouts", drop1)

    # second volutional layer
    with tf.name_scope('Conv2'):
        w2 = tf.Variable(tf.truncated_normal([6, 4, 44, 44], stddev=0.01), name='W')
        b2 = tf.Variable(tf.zeros([44]), name='B')
        conv2 = tf.nn.conv2d(max_pool1, w2, strides=[1, 1, 1, 1], padding='SAME')
        act2 = tf.nn.relu(conv2 + b2)
        drop2 = tf.nn.dropout(act2, dropout)
        tf.summary.histogram("weights", w2)
        tf.summary.histogram("biases", b2)
        tf.summary.histogram("activations", act2)
        tf.summary.histogram("dropouts", drop2)

        # reshaping for fc layer
        conv_shape = drop2.get_shape()
        count = int(conv_shape[1] * conv_shape[2] * conv_shape[3])
        flat_output = tf.reshape(drop2, [-1, count])

    # fc layer
    with tf.name_scope('FC'):
        w3 = tf.Variable(tf.truncated_normal([count, NUM_LABELS], stddev=0.01))
        b3 = tf.Variable(tf.zeros([NUM_LABELS]))
        fc = tf.add(tf.matmul(flat_output, w3), b3)
        tf.summary.histogram("weights", w3)
        tf.summary.histogram("biases", b3)

    return fc

def main():
    tf.reset_default_graph()
    sess = tf.Session()

    x = tf.placeholder(tf.float32, shape=[None, HEIGHT, WIDTH], name="input")
    y = tf.placeholder(tf.float32, shape=[None, NUM_LABELS], name="label")
    dropout = tf.placeholder(tf.float32, name='dropout')

    # NN model
    logits = get_model(x, dropout)
    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
        tf.summary.scalar("loss", loss)
    with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)
    with tf.name_scope('accuracy'):
        predicted = tf.argmax(logits, 1)
        truth = tf.argmax(y, 1)
        correct_predict = tf.equal(predicted, truth)
        accuracy = tf.reduce_mean(tf.cast(correct_predict, tf.float32))
        confusion_matrix = tf.confusion_matrix(truth, predicted, num_classes=NUM_LABELS)
        tf.summary.scalar('accuracy', accuracy)

    # tensorboard
    summ = tf.summary.merge_all()
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(LOGDIR)
    writer.add_graph(sess.graph)
    test_writer = tf.summary.FileWriter(TEST_LOGDIR)

    # model training
    print('Start training\n')
    batch = get_batch(BATCH_SIZE, PATH_TRAIN)
    start_time = time.time()
    for i in range(1, ITERATIONS + 1):
        X, Y = next(batch)
        if i % EVAL_EVERY == 0:
            [train_accuracy, train_loss, s] = sess.run([accuracy, loss, summ], feed_dict={x:X, y:Y, dropout:0.4})
            acc_and_loss = [i, train_loss, train_accuracy * 100]
            print('Iteration # {}. Train Loss: {:.2f}. Train Acc: {:.0f}%'.format(*acc_and_loss))
            writer.add_summary(s, i)
        if i % (EVAL_EVERY * 20) == 0:
            train_confusion_matrix = sess.run([confusion_matrix], feed_dict={x:X, y:Y, dropout:0.5})
            header = LABEL_TO_INDEX_MAP.keys()
            df = pd.DataFrame(np.reshape(train_confusion_matrix, (NUM_LABELS, NUM_LABELS)))
            print('\nConfusion Matrix:\n {}\n '.format(df))
            saver.save(sess, os.path.join(LOGDIR, "model.ckpt"), i)

        sess.run(train_step, feed_dict={x:X, y:Y, dropout:0.5})

    print('\nTotal training time: {:0f} seconds\n'.format(time.time() - start_time))

    # model testing
    batch = get_batch(BATCH_SIZE, PATH_TEST)
    total_accuracy = 0
    for i in range(ITERATIONS_TEST):
        X, Y = next(batch, PATH_TEST)
        test_accuracy, s = sess.run([accuracy, summ], feed_dict={x:X, y:Y, dropout:1.0})
        print('Iteration # {}. Test Accuracy: {:.0f}%'.format(i + 1, test_accuracy * 100))
        total_accuracy += (test_accuracy / ITERATIONS_TEST)
        test_writer.add_summary(s, i)

    print('\nFinal Test Accuracy: {:.0f}%'.format(total_accuracy * 100))

if __name__ == '__main__':
    init(PATH_TRAIN)
    main()