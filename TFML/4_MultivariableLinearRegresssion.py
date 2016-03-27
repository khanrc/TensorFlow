# coding: utf-8

#coding:utf-8

import tensorflow as tf
import numpy as np

# x1_data = [1, 0, 3, 0, 5]
# x2_data = [0, 2, 0, 4, 0]
# matrix 로 변환
# b 를 weights W 안으로 집어넣기
# x_data = [[1, 1, 1, 1, 1], # b를 위한 상수 1
#           [1, 0, 3, 0, 5],
#           [0, 2, 0, 4, 0]]
# y_data = [1, 2, 3, 4, 5]

# numpy 를 이용해서 train.txt 로부터 데이터를 불러올 수 있다.
xy = np.loadtxt('train.txt', unpack=True, dtype='float32')
# print(xy)
x_data = xy[0:-1]
y_data = xy[-1]

# W1 = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
# W2 = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
# matrix 로 변환
W = tf.Variable(tf.random_uniform([1, len(x_data)], -1.0, 1.0))
# b = tf.Variable(tf.random_uniform([1], -1.0, 1.0)) # b 는 W안으로 집어넣음

# matrix
X = tf.placeholder(tf.float32)
# X1 = tf.placeholder(tf.float32)
# X2 = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# predict = W1*X1 + W2*X2 + b
# matrix
predict = tf.matmul(W,X) # b 를 더할 필요가 없어짐
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
    for step in xrange(2001):
        sess.run(train, feed_dict={X: x_data, Y: y_data})
        if step % 20 == 0:
            print step, sess.run(cost, feed_dict={X:x_data, Y:y_data}), sess.run(W)
