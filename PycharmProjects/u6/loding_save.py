import tensorflow as tf
import numpy as np
n_input=784
n_classes=10
x=tf.placeholder('float',[None,n_input])
y=tf.placeholder('float',[None,n_classes])
saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess,"model/model_time20.ckpt")
    print sess.run(accuracy,feed_dict=feeds)