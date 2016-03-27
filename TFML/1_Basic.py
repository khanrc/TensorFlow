import tensorflow as tf

a = tf.placeholder(tf.int16)
b = tf.placeholder(tf.int16)

# models
add = tf.add(a, b)
mul = tf.mul(a, b)
m2 = a*b

with tf.Session() as sess:
    print sess.run(add, feed_dict={a:2, b:3})
    print sess.run(mul, feed_dict={a:[2,3], b:[3,4]})
    print sess.run(m2, feed_dict={a:[2,3], b:[3,4]})