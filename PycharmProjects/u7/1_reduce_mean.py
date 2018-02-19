import tensorflow as tf
import numpy as np
m=tf.Variable(tf.zeros([4,4])+1,tf.float32)
c=tf.reduce_mean(m)
init=tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    print sess.run(c)