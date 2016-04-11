# coding: utf-8

import tensorflow as tf
import numpy as np

# numpy 를 이용해서 train.txt 로부터 데이터를 불러올 수 있다.
xy = np.loadtxt('train.txt', unpack=True, dtype='float32')
x_data = xy[0:-1]
y_data = xy[-1]

print "x data:", x_data
print "y data:", y_data

W = tf.Variable(tf.random_uniform([1, len(x_data)], -1.0, 1.0))

# matrix
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

h = tf.matmul(W, X)
predict = tf.div(1., 1.+tf.exp(-h))
# cost = tf.reduce_mean(tf.square(predict-Y))
cost = -tf.reduce_mean(Y*tf.log(predict) + (1-Y)*tf.log(1-predict))

a = tf.Variable(0.2) # learning rate
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

# init vars
init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    for step in xrange(2001):
        sess.run(train, feed_dict={X: x_data, Y: y_data})
        if step % 50 == 0:
            print step, sess.run(cost, feed_dict={X:x_data, Y:y_data}), sess.run(W)

    print sess.run(predict, feed_dict={X:np.matrix([1,1,5]).T})
    w = sess.run(W)[0]

# plot train data
import matplotlib.pyplot as plt
print(x_data[1], x_data[2])
plt.plot(x_data[1][:4], x_data[2][:4], 'bo')
plt.plot(x_data[1][4:], x_data[2][4:], 'ro')
plt.xlim([0,10])
plt.ylim([0,10])

# plot decision boundary
x = np.arange(0, 10, 0.1)
y = (-w[1]*x - w[0] + 0.5) / w[2]
plt.plot(x,y)
plt.show()