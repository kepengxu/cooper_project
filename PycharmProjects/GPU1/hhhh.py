# -*- coding: UTF-8 -*-
import tensorflow as tf
import sys
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import random
x=(np.arange(0,50)+1960).reshape(1,50).astype(np.float32)
data=np.loadtxt('/home/cooper/Downloads/MCM/emer/AZ.csv',skiprows=1,delimiter=',')
total=data[:,-2]
rec=data[:,-3]
y=(rec/total).reshape(1,50)

#x=tf.placeholder(tf.float32,shape=[1,50])
#y=tf.placeholder(tf.float32,shape=[1,50])
x2=(np.arange(0,50)+2001).reshape(1,50).astype(np.float32)
istrain=1
inputdim=50
outputdim=50

def add_layer(input,inputsize,outputsize,layername,activate_function=None):
    with tf.name_scope(layername):
        with tf.name_scope('weight'):
            weight=tf.Variable(tf.random_normal([inputsize,outputsize],0.1,1.0),name='weight')
#   if don't use batch ,needn't to use batch normalization
        with tf.name_scope('bias'):
            bias=tf.Variable(tf.zeros([outputsize]))
    output=tf.add(tf.matmul(input,weight),bias)
    if activate_function is None:
        output=output
    else:
        output=activate_function(output)
    return output

l1=add_layer(x,inputdim,256,'11',tf.nn.relu)
l2=add_layer(l1,256,512,'22',tf.nn.relu)
l3=add_layer(l2,512,outputdim,'33',tf.nn.relu)

loss=tf.reduce_mean((l3-y)**2)
op=tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
init=tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    if istrain==1:
        sess.run(op)
        for i in range(100000):
            sess.run(op)
            if i%1000==0:
                los=sess.run(loss)
                print los
            if i%10000==0:
                print 'trin is over'
                x=x+41
                break
    sa=sess.run(l3)
    print sa

    plt.plot(x, sa, color="blue", linewidth=1, linestyle="-", label='Arizona')
    plt.show()
