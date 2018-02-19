# -*- coding: UTF-8 -*-
import tensorflow as tf
import sys
import numpy as np
from sklearn.decomposition import PCA
import random
wether_train=0

batch_size=None
input_dim=1
output_dim=1
x=tf.placeholder(tf.float32,[50,input_dim],name='input')
y=tf.placeholder(tf.float32,[50,output_dim],name='stand_output')
#phase_train=tf.placeholder(tf.bool,name='phase_train')
keep_prob=tf.placeholder(tf.float32,name='keep_prob')
learning_rate=0.001
#xs,pca_x=load_x()
xs=(np.arange(0,50)+1960).reshape(50,1).astype(np.float32)
xsm=tf.constant(xs,tf.float32)
xss=(np.arange(0,50)+1960).reshape(50,1).astype(np.float32)
data=np.loadtxt('/home/cooper/Downloads/MCM/emer/AZ.csv',skiprows=1,delimiter=',')
total=data[:,-2]
rec=data[:,-3]
ys=(rec/total).reshape(50,1)
ysm=tf.constant(ys,tf.float32)
def add_layer(input,inputsize,outputsize,layername,activate_function=None):
    with tf.name_scope(layername):
        with tf.name_scope('weight'):
            weight=tf.Variable(tf.random_normal([inputsize,outputsize],0.1,1.0),name='weight')
#   if don't use batch ,needn't to use batch normalization
        with tf.name_scope('bias'):
            bias=tf.Variable(tf.zeros([outputsize]))
    output=tf.add(tf.matmul(input,weight),bias)
    output=tf.nn.dropout(output,keep_prob)
    if activate_function is None:
        output=output
    else:
        output=activate_function(output)
    return output
def inference(x,input_dim,output_dim):
    with tf.name_scope('layer_1'):
        l1 = add_layer(x, input_dim, 128, layername='layer_1', activate_function=tf.nn.relu)
    with tf.name_scope('layer_2'):
        l2 = add_layer(l1, 128, 256, layername='layer_2', activate_function=tf.nn.sigmoid)
    with tf.name_scope('layer_3'):
        l3 = add_layer(l2, 256, 512, layername='layer_3', activate_function=tf.nn.relu)
    with tf.name_scope('layer_4'):
        l4 = add_layer(l3, 512, 256, layername='layer_4', activate_function=tf.nn.relu)
    with tf.name_scope('layer_5'):
        l5 = add_layer(l4, 256, 128, layername='layer_5', activate_function=tf.nn.sigmoid)
    with tf.name_scope('layer_6'):
        l6= add_layer(l5, 128, 128, layername='layer_6', activate_function=tf.nn.sigmoid)
    with tf.name_scope('layer_7'):
        l7 = add_layer(l6, 128, output_dim, layername='layer_7', activate_function=tf.nn.sigmoid)
    pre=l7
    return pre
pre=inference(xs,input_dim,output_dim)
cross_entropy =tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y,logits=pre))
loss=tf.reduce_mean((pre-y)**2)
train_op=tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
init=tf.global_variables_initializer()
saver=tf.train.Saver()
with tf.Session() as sess:
    train_writer = tf.summary.FileWriter("MBM/train", sess.graph)
    sess.run(init)
    if wether_train:
        for i in range(100000):
            sess.run(train_op,feed_dict={x:xs,y:ys,keep_prob:1})
            if i%10000==0:
                los=sess.run(loss,feed_dict={x:xs,y:ys,keep_prob:1})
                saver.save(sess, '/home/cooper/PycharmProjects/GPU1/MBM/save_sess')
                print los
                sys.exit(0)

#            cross_entropy=sess.run(cross_entropy,feed_dict={x:xs,y:ys,keep_prob:1})

    else:
        saver.restore(sess, '/home/cooper/PycharmProjects/GPU1/MBM/save_sess')
        pr=sess.run(pre,feed_dict={x:xss,y:ys,keep_prob:1})
        print pr
    print ys

