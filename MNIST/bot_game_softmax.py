#coding: utf-8

import input_data
import tensorflow as tf
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

# 1. softmax: 0.914
def weight_variable(shape, name=None):
    initial = tf.truncated_normal(shape, stddev=0.1, name=name)
    # initial = tf.zeros(shape)
    return tf.Variable(initial)

def bias_variable(shape, name=None):
    initial = tf.constant(0.01, shape = shape, name=name)
    # initial = tf.zeros(shape)
    return tf.Variable(initial)

def create_network_softmax():
    # weights
    w_softmax = weight_variable([784, 10])
    b_softmax = bias_variable([10])

    # input, target
    x = tf.placeholder("float", [None, 28*28])
    y = tf.placeholder("float", [None, 10])

    # readout layer (softmax)
    readout = tf.matmul(x, w_softmax) + b_softmax

    return x, y, readout

def train_network(x, y, readout):
    predict = tf.nn.softmax(readout)
    learning_rate = 1e-2
    # cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(predict), reduction_indices=1))
    cost = -tf.reduce_sum(y*tf.log(predict)) # cross-entropy

    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

    return cost, optimizer

if __name__ == "__main__":
    x, y, readout = create_network_softmax()
    cost, optimizer = train_network(x, y, readout)

    init = tf.initialize_all_variables()

    with tf.Session() as sess:
        sess.run(init)

        for step in xrange(1000):
            batch = mnist.train.next_batch(50)
            sess.run(optimizer, feed_dict={x: batch[0], y:batch[1]})

            if step % 100 == 0:
                print step, sess.run(cost, feed_dict={x:batch[0], y:batch[1]})

        # tf.argmax: second variable indicates 'axis'.
        # one iteration find argmax in 0 => column, 1 => row.
        result = sess.run(tf.argmax(readout,1), feed_dict={x: mnist.validation.images})
        correct = tf.equal(tf.argmax(mnist.validation.labels, 1), result)
        accuracy = tf.reduce_mean(tf.cast(correct, "float"))
        print sess.run(accuracy)


    # print ' '.join(map(str, result))