#coding:utf-8

__author__ = 'CJB'

import input_data
mnist = input_data.read_data_sets("data/", one_hot=True)

import tensorflow as tf
import numpy as np


def xavier_init(n_inputs, n_outputs, uniform=True):
    if uniform:
        # 6 was used in the paper.
        init_range = np.sqrt(6.0 / (n_inputs + n_outputs))
        return tf.random_uniform_initializer(-init_range, init_range)
    else:
        # 3 gives us approximately the same limits as above since this repicks
        # values greater than 2 standard deviations from the mean.
        stddev = np.sqrt(3.0 / (n_inputs + n_outputs))
        return tf.truncated_normal_initializer(stddev=stddev)

def xavier_variable(shape, name):
    return tf.get_variable(name, shape=shape, initializer=xavier_init(shape[0], shape[1]))

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.01, shape = shape)
    return tf.Variable(initial)

learning_rate = 0.001
training_epochs = 15
batch_size = 100
display_step = 1

# training_epochs 를 15정도로 하고 5단으로 했을 때, dropout을 적용하는 것이 0.4%정도 좋은 결과를 낸다.

x = tf.placeholder("float", [None, 784])
y = tf.placeholder("float", [None, 10])
keep_prob = tf.placeholder("float")

# W1 = tf.Variable(tf.zeros([784, 256]))
# W2 = tf.Variable(tf.zeros([256, 256]))
# W3 = tf.Variable(tf.zeros([256, 10]))
# b = tf.Variable(tf.zeros([10]))

# W1 = weight_variable([784, 256])
# W2 = weight_variable([256, 256])
# W3 = weight_variable([256, 128])
# W4 = weight_variable([128, 128])
# W5 = weight_variable([128, 10])

W1 = xavier_variable([784, 256], "W1")
W2 = xavier_variable([256, 256], "W2")
W3 = xavier_variable([256, 128], "W3")
W4 = xavier_variable([128, 128], "W4")
W5 = xavier_variable([128, 10], "W5")

b1 = bias_variable([256])
b2 = bias_variable([256])
b3 = bias_variable([128])
b4 = bias_variable([128])
b5 = bias_variable([10])

L1 = tf.nn.relu(tf.matmul(x, W1) + b1)
L1 = tf.nn.dropout(L1, keep_prob)
L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)
L2 = tf.nn.dropout(L2, keep_prob)
L3 = tf.nn.relu(tf.matmul(L2, W3) + b3)
L3 = tf.nn.dropout(L3, keep_prob)
L4 = tf.nn.relu(tf.matmul(L3, W4) + b4)
L4 = tf.nn.dropout(L4, keep_prob)

# activation = tf.nn.softmax(tf.matmul(x, W) + b)
activation = tf.matmul(L4, W5) + b5

# cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(activation), reduction_indices=1))
# cost = -tf.reduce_sum(y*tf.log(activation)) # 같은 cross-entropy 인데 이게 더 학습이 잘 됨.
# => 그 이유는 이건 그냥 sum 이고 위에는 mean 이라서 이게 더 learning_rate 이 큰 효과를 낳는다. 위에꺼 (mean) 써도 learning_rate 을 올리면 학습 잘 됨
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(activation, y)) # 이건 1번이랑 똑같음. 다만 activation 에서 softmax 를 빼줘야함
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# run
init = tf.initialize_all_variables()

print "# train: {}".format(mnist.train.num_examples)
print "# test: {}".format(mnist.test.num_examples)

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples/batch_size)
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 0.7})
            avg_cost += sess.run(cost, feed_dict={x: batch_xs, y:batch_ys, keep_prob: 1.0})/total_batch

        if epoch % display_step == 0:
            print "Epoch:", "%04d" % (epoch+1), "cost=", "{:.9f}".format(avg_cost)

    print "Optimization Finished!"

    # Test
    correct_prediction = tf.equal(tf.argmax(activation, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print "Accuracy:", accuracy.eval({x: mnist.test.images, y:mnist.test.labels, keep_prob: 1.0})

    # import random
    # r = random.randint(0, mnist.test.num_examples - 1)
    # print "Label: ", sess.run(tf.argmax(mnist.test.labels[r:r+1], 1))
    # print "Prediction: ", sess.run(tf.argmax(activation, 1), feed_dict={x: mnist.test.images[r:r+1]})
    #
    # import matplotlib.pyplot as plt
    # plt.imshow(mnist.test.images[r:r+1].reshape(28, 28), cmap='Greys', interpolation='nearest')
    # plt.show()
