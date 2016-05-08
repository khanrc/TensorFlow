__author__ = 'CJB'

import tensorflow as tf
from tensorflow.models.rnn import rnn, rnn_cell
import numpy as np

char_rdic = ['h', 'e', 'l', 'o']
char_dic = {w: i for i, w in enumerate(char_rdic)} # char -> id
x_data = np.array([[1,0,0,0],
                   [0,1,0,0],
                   [0,0,1,0],
                   [0,0,1,0]], dtype=np.float32)

sample = [char_dic[c] for c in "hello"] # indexed_data

# config
char_vocab_size = len(char_dic) # 4
state_size = char_vocab_size # one-hot coding (1 of 4)
batch_size = 1 # one input (one output) per single time step
time_step_size = char_vocab_size / batch_size # 1:1 predict. hell => ello

print(state_size, time_step_size)

# RNN model
cell = rnn_cell.BasicRNNCell(state_size)
init_state = tf.zeros([batch_size, cell.state_size]) # initial state
X_split = tf.split(0, time_step_size, x_data) # split data fitting into batch_size
outputs, state = rnn.rnn(cell, X_split, init_state)

# what's difference between state_size and cell.state_size?
logits = tf.reshape(tf.concat(1, outputs), [-1, state_size])
targets = tf.reshape(sample[1:], [-1])
weights = tf.ones([time_step_size * batch_size])

loss = tf.nn.seq2seq.sequence_loss_by_example([logits], [targets], [weights])
cost = tf.reduce_sum(loss)/batch_size
train_op = tf.train.RMSPropOptimizer(0.01, 0.9).minimize(cost)

# Launch
with tf.Session() as sess:
    tf.initialize_all_variables().run()
    for i in range(100):
        sess.run(train_op)
        logits_res = sess.run(logits)
        result = sess.run(tf.arg_max(logits, 1))
        print(logits_res)
        print(result, [char_rdic[t] for t in result])
