__author__ = 'CJB'

import numpy as np
import tensorflow as tf

xy = np.loadtxt('6_train.txt', unpack=True, dtype='float32')
x_data = xy[0:3].T
y_data = xy[3:].T

print(x_data)
print(y_data)

X = tf.placeholder("float", [None, 3]) # x1, x2, bias
Y = tf.placeholder("float", [None, 3]) # A, B, C (3 classes)
W = tf.Variable(tf.zeros([3, 3])) # no init?

predict = tf.nn.softmax(tf.matmul(X, W)) # softmax
learning_rate = 0.01
cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(predict), reduction_indices=1))
# cost = -tf.reduce_sum(Y*tf.log(predict))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    for step in xrange(2001):
        sess.run(optimizer, feed_dict={X: x_data, Y: y_data})
        if step % 100 == 0:
            print step, sess.run(cost, feed_dict={X:x_data, Y:y_data})#, sess.run(W)

    # all = sess.run(predict, feed_dict={X:[[1, 11, 7], [1, 3, 4], [1, 1, 0]]})
    # print all, sess.run(tf.arg_max(all, 1))


