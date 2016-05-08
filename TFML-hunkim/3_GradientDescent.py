# coding:utf-8
# GD를 직접 구현해 보자.

import tensorflow as tf

x_data = [1., 2., 3.]
y_data = [1., 2., 3.]

W = tf.Variable(tf.random_uniform([1], -10., 10.))

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

predict = W*X
cost = tf.reduce_mean(tf.square(predict - Y))

# GD
descent = W-tf.mul(0.1, tf.reduce_mean((W*X - Y)*X))
update = W.assign(descent) # update operation. W에 할당된 값을 리턴함

init = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)
    for step in xrange(20):
        sess.run(update, feed_dict={X:x_data, Y:y_data})
        print step, sess.run(cost, feed_dict={X:x_data, Y:y_data}), sess.run(W)
