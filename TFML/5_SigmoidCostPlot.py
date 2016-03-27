# coding: utf-8
# sigmoid cost function: H(X) = 1/(1+e^{-W^T*X}).

import tensorflow as tf

X = [1., 2., 3., 4.]
Y = [0., 0., 1., 1.]
N = len(X)

W = tf.placeholder(tf.float32)
h = tf.mul(W, X)
predict = tf.div(1., 1.+tf.exp(-h))

# cost = tf.reduce_mean(tf.abs(predict-Y))
# cost = tf.reduce_mean(tf.pow(predict-Y, 2))
cost = -tf.reduce_mean(tf.log(predict) + tf.sub(1., Y)*tf.log(1-predict))
# 이렇게 log 가 들어가야 코스트 펑션이 컨벡스 펑션이 됨.

init = tf.initialize_all_variables()

W_val = []
cost_val = []
with tf.Session() as sess:
    sess.run(init)
    # print sess.run(tf.sub(1., Y)*X)
    for i in range(-50, 50):
        print i*0.1, sess.run(cost, feed_dict={W: i*0.1}), sess.run(predict, feed_dict={W: i*0.1})
        W_val.append(i*0.1)
        cost_val.append(sess.run(cost, feed_dict={W: i*0.1}))

# exit(1)

# plot
from matplotlib import pyplot as plt
plt.plot(W_val, cost_val, 'ro')
plt.ylabel('Cost')
plt.xlabel('W')
plt.show()