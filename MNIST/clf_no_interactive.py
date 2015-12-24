# coding=utf-8
import tensorflow as tf
import input_data


# download data
# mnist is traning/validation/test set as Numpy array
# Also it provides a function for iterating through data minibatches
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# InteractiveSession을 쓰지 않으면 세션에 그래프를 올리기 전에 그래프를 전부 완성해야 함.
# 다른말로 하면 계산하기 전에 그래프를 완성해야 함. InteractiveSession을 쓰면 그때그때 계산이 가능함.
sess = tf.InteractiveSession()

# 파이썬에서 연산을 빠르게 하기 위해 numpy와 같은 라이브러리들은 파이썬 바깥 (아마도 C++) 에서 연산을 수행하게 한다.
# 그러나 이런 노력에도 불구하고 이러한 switching 자체가 오버헤드임. 특히 GPU나 분산처리할 때 심각함.
# 이러한 문제를 해결하기 위해 텐서플로는 연산 하나하나를 C++로 수행하는게 아니라 연산 전체를 통째로 C++로 수행함.
# 이러한 접근방식은 Theano나 Torch와 유사
# 파이썬의 역할은 전체 컴퓨테이션 그래프를 구축하고, 어느 부분이 실행되어야 하는지를 명시하는 일이다.

# 즉 그래프를 만들고 run()을 통해 원하는 부분을 실행하면 해당 부분이 Session에 올려져 실행이 되는데 이 작업이 통째로 외부에서 실행되는듯.
# 그렇기 때문에 원래는 InteractiveSession을 쓰면 안 되는거임. InteractiveSession을 쓰면 속도가 상당히 느려지리라 짐작함.

# 본문에서는 파이썬 바깥 혹은 파이썬으로부터 독립적으로 연산한다고 표현하므로 정확히 C++인지는 잘 모르겠음.


# Softmax Regression Model
# 1-layer NN => Softmax. 1-layer란 Input layer와 Output layer만 있는 것이 1-layer임.
# placeholder는 실행할때 우리가 값을 넣을 수 있음
x = tf.placeholder("float", shape=[None, 784])  # x는 input image. 784 = 28*28, 이미지를 핌 (flatten). 흑백 이미지이므로 각 픽셀은 0/1
y_ = tf.placeholder("float", shape=[None, 10])  # _y는 class label. mnist가 0~9까지의 이미지이므로 10개. one-hot 벡터.

# Variables: Weights & Bias
# Variable은 말 그대로 변수에 해당한다.
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# 변수는 사용하기 전에 초기화해줘야 한다. 선언시에 정해준 값으로 (여기서는 0) 초기화된다.
sess.run(tf.initialize_all_variables())  # 모든 변수 초기화

# Predicted Class and Cost Function
# softmax 함수는 이미 구현되어 있으므로 한줄에 짤 수 있음.
# tf.nn.softmax는 소프트맥스 함수만을 말하고 이 과정 전체가 소프트맥스 리그레션이다.
y = tf.nn.softmax(tf.matmul(x, W) + b) # Wx+b = Output nodes의 액티베이션 값. 즉 액티베이션 값을 소프트맥스 함수에 넣음.

# Cost function: 트레이닝 과정에서 최소화해야 하는 값.
# cross-entropy between the target and the model's prediction.
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
# reduce_sum은 텐서의 합을 구함. reduce는 텐서를 축소한다는 개념인 듯.
"""
```python
# 'x' is [[1, 1, 1]]
#         [1, 1, 1]]
tf.reduce_sum(x) ==> 6
tf.reduce_sum(x, 0) ==> [2, 2, 2]
tf.reduce_sum(x, 1) ==> [3, 3]
tf.reduce_sum(x, 1, keep_dims=True) ==> [[3], [3]]
tf.reduce_sum(x, [0, 1]) ==> 6
```
"""

# Train the Model
# 지금까지, 우리의 모델과 코스트 펑션을 정의했음.
# 즉 다시 말해서 텐서플로는 우리의 전체 컴퓨테이션 그래프를 알고 있음.
# 이에 기반하여 자동으로 미분하여 (differentiation) gradient를 계산할 수 있음.
# 텐서플로는 다양한 최적화 알고리즘을 제공함. http://www.tensorflow.org/api_docs/python/train.html#optimizers
# 여기서는 steepest gradient descent를 사용.
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
# 이 한 줄은 그래프에 하나의 op를 추가한 것. 이 op는 gradient를 계산하고, 파라메터 업데이트 스텝을 계산하고, 그 결과를 파라메터에 적용한다.
# 즉 위 스텝을 반복하면 점점 학습이 됨.

'''
# 50 크기의 mini-batch로 1000번 학습을 함.

for i in range(1000):
    batch = mnist.train.next_batch(50) # mini-batch (50)
    # batch[0]은 x, batch[1]은 y로 구성됨. [data, class_label] 구조.
    # batch[0 or 1][0~49] 가 각각의 데이터.
    # batch_xs, batch_ys = mnist.train.next_batch(50) 형태로 받는 것이 더 직관적.
    train_step.run(feed_dict={x: batch[0], y_:batch[1]})
    y_eval = y.eval(feed_dict={x:batch[0]})


# Evaluate the Model
# argmax는 tensor에서 max값을 찾는 함수다. 파라메터는 tensor, axis (dimension) 임. axis개념은 numpy와도 같고 위의 reduce_sum과도 같음.
# reduce_mean도 마찬가지.
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))  # 예측한 라벨과 정답 라벨을 비교하고
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))  # 비교한 결과의 평균을 낸다
print accuracy.eval(feed_dict={x: mnist.test.images, y_:mnist.test.labels})  # 실행

# => 0.9092

# 참고: Tensor.eval(), Operation.run(). (in InteractiveSession)


exit(1)
'''
# ---------------------------------------------------------------------------------------------------------------------#

# CNN

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


# convolution & max pooling
# vanila version of CNN
# x (아래 함수들에서) : A 4-D `Tensor` with shape `[batch, height, width, channels]`
def conv2d(x, W):
    # stride = 1, zero padding은 input과 output의 size가 같도록.
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# First convolutional layer
# [5, 5, 1, 32]: 5x5 convolution patch, 1 input channel, 32 output channel.
# MNIST의 pixel은 0/1로 표현되는 1개의 벡터이므로 1 input channel임.
# CIFAR-10 같이 color인 경우에는 RGB 3개의 벡터로 표현되므로 3 input channel일 것이다.
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
# 최종적으로, 32개의 output channel에 대해 각각 5x5의 convolution patch (filter) weight 와 1개의 bias 를 갖게 됨.

# x는 [None, 784] (위 placeholder에서 선언). 이건 [batch, 28*28] 이다.
# x_image는 [batch, 28, 28, 1] 이 됨. -1은 batch size를 유지하는 것이고 1은 color channel.
x_image = tf.reshape(x, [-1,28,28,1])

# 이제, x_image를 weight tensor와 convolve하고 bias를 더한 뒤 ReLU를 적용하자. 그리고 마지막으론 max pooling.
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)



# Second convolutional layer
# 5x5x32x64 짜리 weights.
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)


# Densely connected layer
# 7*7*64는 h_pool2의 output (7*7의 reduced image * 64개의 채널). 1024는 fc layer의 뉴런 수.
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64]) # -1은 batch size를 유지하는 것.
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)


# Dropout
# keen_prob은 dropout을 적용할지 말지에 대한 확률임. 이를 이용해서 training 동안만 드롭아웃을 적용하고 testing 때는 적용하지 않는다.
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


# Readout layer
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)




cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
sess.run(tf.initialize_all_variables())
for i in range(20000):
    batch = mnist.train.next_batch(50)
    if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
        print "step %d, training accuracy %g" % (i, train_accuracy)
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print "test accuracy %g" % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})

