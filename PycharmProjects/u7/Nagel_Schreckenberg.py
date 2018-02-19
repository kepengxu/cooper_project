import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
x=tf.Variable(tf.random_uniform([1,100],1,100,tf.int16))
y=tf.Variable(tf.random_uniform([1,100],1,100,tf.int16))
mat=tf.Variable([1000,100])
with tf.Session() as sess:
    