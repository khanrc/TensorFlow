# coding:utf-8
# cost function 을 그려 보자.

import tensorflow as tf

X = [1., 2., 3., 4., 5., 6., 7., 8., 9.]
Y = [10., 20., 30., 10., 10., 10., 7., 8., 9.]
N = len(X)

W = tf.placeholder(tf.float32)
predict = tf.mul(X, W)
# predict = X*W # same same

cost = tf.reduce_sum(tf.pow(predict-Y, 2))/N

init = tf.initialize_all_variables()

W_val = []
cost_val = []
with tf.Session() as sess:
    sess.run(init)
    for i in range(-30, 50):
        print i*0.1, sess.run(cost, feed_dict={W: i*0.1})
        W_val.append(i*0.1)
        cost_val.append(sess.run(cost, feed_dict={W: i*0.1}))

# plot
from matplotlib import pyplot as plt
plt.plot(W_val, cost_val, 'ro')
plt.ylabel('Cost')
plt.xlabel('W')
plt.show()