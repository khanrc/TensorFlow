# coding=utf-8
import tensorflow as tf
import numpy as np


def read_data():
    ret = []
    with open("wisconsin-original-data.txt") as f:
        for line in f:
            row = line.strip().split(',')
            if '?' in row:
                # print(row)
                continue
            ret.append(row)
    return ret

def dense_to_onehot(labels_dense):
    l1 = (labels_dense == 2)
    l2 = (labels_dense == 4)
    labels_onehot = np.array([l1, l2]).T
    return labels_onehot.astype(int)

# read data
data = np.array(read_data())
data = data.astype(int)

size = len(data)

train_size = int(size * 0.5)

train_data = data[:train_size]
test_data = data[train_size:]
# print(len(train_data))
# print(len(test_data))

train_features = train_data[:, 1:-1]
train_labels = dense_to_onehot(train_data[:, -1])
test_features = test_data[:, 1:-1]
test_labels = dense_to_onehot(test_data[:, -1])


# TensorFlow - Autoencoders
sess = tf.InteractiveSession()

# truncated normal distribution에 기반해서 랜덤한 값으로 초기화
def weight_variable(shape):
    # tf.truncated_normal:
    # Outputs random values from a truncated normal distribution.
    # values whose magnitude is more than 2 standard deviations from the mean are dropped and re-picked.
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

# 0.1로 초기화
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# autoencoder
# 오토인코더는 fully-connected network + 1 channel.
# hidden layer 1 : 4
# hidden layer 2 : 2
# softmax
# dropout은 일단 생략

# placeholder
# x is input, y_ is class label (output). None is batch size
x = tf.placeholder("float", shape=[None, 9])  
y_ = tf.placeholder("float", shape=[None, 2])  

# first layer
# x: input (features of breast cancer)
W1 = weight_variable([9, 4])
b1 = bias_variable([4])

h1 = tf.nn.relu(tf.matmul(x, W1) + b1)
# h1 = tf.nn.sigmoid(tf.matmul(x, W1) + b1)

# second layer (readout layer - softmax)
W2 = weight_variable([4, 2])
b2 = bias_variable([2])

# h2 = tf.nn.relu(tf.matmul(h1, W2) + b2)
y = tf.nn.softmax(tf.matmul(h1, W2) + b2)

# MLE (negative log-likelihood)
cross_entropy = -tf.reduce_sum(y_*tf.log(y))

train_step = tf.train.GradientDescentOptimizer(1e-4).minimize(cross_entropy)
# train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy) # 뭐가 더 좋은지 모르겠음

# accuracy calculation
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# run
# no mini-batch. shotgun gradient descent (because dataset size is small)
# no dropout => no keen prob
sess.run(tf.initialize_all_variables())
for i in range(5000):
    if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x:train_features, y_: train_labels})
        print "step %d, training accuracy %g" % (i, train_accuracy)
    train_step.run(feed_dict={x: train_features, y_: train_labels})

print "test accuracy %g" % accuracy.eval(feed_dict={x: test_features, y_: test_labels})



# for i in range(2000):
#     batch = mnist.train.next_batch(50)
#     if i%100 == 0:
#         train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
#         print "step %d, training accuracy %g" % (i, train_accuracy)
#     train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

# print "test accuracy %g" % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})

