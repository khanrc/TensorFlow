#coding:utf-8

import tensorflow as tf

x_data = [1,2,3]
y_data = [0.4,0.5,0.6]

W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

predict = W*X + b
cost = tf.reduce_mean(tf.square(predict-Y))
# reduce_mean 은 tensor 의 평균을 구하는데 텐서의 크기를 축소한다.
# 들어가면 나오는 예제 참고.

a = tf.Variable(0.1) # learning rate
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

# init vars
init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    for step in xrange(1001):
        sess.run(train, feed_dict={X: x_data, Y: y_data})
        if step % 20 == 0:
            print step, sess.run(cost, feed_dict={X: x_data, Y:y_data}), sess.run(W), sess.run(b)

    print sess.run(predict, feed_dict={X: 5})