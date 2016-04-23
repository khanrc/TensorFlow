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

# def create_network_deep():
# weights
w1 = weight_variable([784, 300], name='Weight1')
w1_hist = tf.histogram_summary("weights1", w1)
b1 = bias_variable([300], name='Bias1')
b1_hist = tf.histogram_summary("biases1", b1)

w2 = weight_variable([300, 100], name='Weight2')
w2_hist = tf.histogram_summary("weights2", w2)
b2 = bias_variable([100], name='Bias2')
b2_hist = tf.histogram_summary("biases2", b2)

w3 = weight_variable([100, 10], name='Weight3')
w3_hist = tf.histogram_summary("weights3", w3)
b3 = bias_variable([10], name='Bias3')
b3_hist = tf.histogram_summary("biases3", b3)

# input, target
x = tf.placeholder("float", [None, 28*28], name='Input')
y = tf.placeholder("float", [None, 10], name='Target')
y_hist = tf.histogram_summary("y", y)

# hidden layer
with tf.name_scope("Hidden-1"):
    h1 = tf.nn.relu(tf.matmul(x, w1) + b1)

# dropout to hidden layer 1
# keep_prob = tf.placeholder("float")
# h1_drop = tf.nn.dropout(h1, keep_prob) # dropout 을 써봤는데 별 차이가 없네.

with tf.name_scope("Hidden-2"):
    h2 = tf.nn.relu(tf.matmul(h1, w2) + b2)

# readout layer (softmax)
with tf.name_scope("Readout"):
    readout = tf.matmul(h2, w3) + b3

    # return x, y, readout, keep_prob

# def train_network(x, y, readout):
with tf.name_scope("Softmax"):
    predict = tf.nn.softmax(readout)
learning_rate = 1e-2
# cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(predict), reduction_indices=1))
with tf.name_scope("Cost"):
    cost = -tf.reduce_sum(y*tf.log(predict)) # cross-entropy
    cost_summ = tf.scalar_summary("Cost", cost)
with tf.name_scope("Optimize"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

    # return cost, optimizer

if __name__ == "__main__":
    # x, y, readout, keep_prob = create_network_deep()
    # cost, optimizer = train_network(x, y, readout)

    init = tf.initialize_all_variables()

    with tf.Session() as sess:
        merged = tf.merge_all_summaries()
        writer = tf.train.SummaryWriter("./logs/mnist_logs", sess.graph_def)
        sess.run(init)

        for step in xrange(1000):
            batch = mnist.train.next_batch(50)
            sess.run(optimizer, feed_dict={x: batch[0], y:batch[1]})

            if step % 100 == 0:
                summary = sess.run(merged, feed_dict={x:batch[0], y:batch[1]})
                writer.add_summary(summary, step)
                print step, sess.run(cost, feed_dict={x:batch[0], y:batch[1]})

        # tf.argmax: second variable indicates 'axis'.
        # one iteration find argmax in 0 => column, 1 => row.
        result = sess.run(tf.argmax(readout,1), feed_dict={x: mnist.validation.images})
        correct = tf.equal(tf.argmax(mnist.validation.labels, 1), result)
        accuracy = tf.reduce_mean(tf.cast(correct, "float"))
        print sess.run(accuracy)


    # print ' '.join(map(str, result))