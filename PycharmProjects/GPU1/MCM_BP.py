# -*- coding: UTF-8 -*-
import tensorflow as tf
import sys
import numpy as np
from sklearn.decomposition import PCA
import random
wether_train=1
def load_x():
    '''
    return two:
    newDate: the data after PCA
    pca:can inverse_transform to org data
    :return:
    '''
    data=np.load('/home/cooper/Downloads/MCM/能源/num.npy')
    print "data shape",data.shape
    pca = PCA(n_components=49, whiten=True)
    newData = pca.fit_transform(data)
    print newData.shape
    print np.sum(pca.explained_variance_ratio_)
    # print pca.explained_variance_ratio_
    M = data
    print "Mshape", M.shape
    inverse = pca.inverse_transform(newData)
    emmm = np.sum(abs(M - inverse))
    return newData,pca
def load_y():
    data = np.loadtxt('/home/cooper/Downloads/MCM/buquan_exp_CSV.csv', delimiter=',', skiprows=2)
    pca = PCA(n_components=49, whiten=True)
    newData = pca.fit_transform(data)
    M = data
    print "data shape", data.shape
    print "Mshape", M.shape
    inverse = pca.inverse_transform(newData)
    emmm = np.sum(abs(M - inverse))
    return newData, pca

batch_size=None
input_dim=49
output_dim=49
x=tf.placeholder(tf.float32,[batch_size,input_dim],name='input')
y=tf.placeholder(tf.float32,[batch_size,output_dim],name='stand_output')
phase_train=tf.placeholder(tf.bool,name='phase_train')
keep_prob=tf.placeholder(tf.float32,name='keep_prob')
#filename=

xs,pca_x=load_x()
ys,pca_y=load_y()
########################need be write


learning_rate=0.0001
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
pre=inference(x,input_dim,output_dim)
with tf.name_scope('loss'):
    cross_entropy =tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y,logits=pre))
    tf.summary.scalar('loss',cross_entropy)
with tf.name_scope('L2_loss'):
    loss=tf.reduce_mean((pre-y)**2)
    tf.summary.scalar('L2_loss', loss)
#numloss=tf.add(loss,(cross_entropy*cross_entropy)/2)
train_op=tf.train.AdamOptimizer(learning_rate).minimize(loss)
saver=tf.train.Saver()
init=tf.global_variables_initializer()
if wether_train==1:
    with tf.Session() as sess:
        sess.run(init)
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter("MCM/train", sess.graph)
        for i in range(100000):
#            j=random.randint(0,49)
            sess.run(train_op,feed_dict={x:xs,y:ys,keep_prob:0.7})
            if i%100==0:
                en_los=sess.run(cross_entropy,feed_dict={x:xs,y:ys,keep_prob:1})
                l2_loss=sess.run(loss,feed_dict={x:xs,y:ys,keep_prob:1})
                if l2_loss<=0.98:
                    saver.save(sess,'/home/cooper/PycharmProjects/GPU1/MCM/save_sess')
                    sys.exit(0)
                print 'cross_entropy_loss',en_los,'l2loss',l2_loss
                train_result=sess.run(merged,feed_dict={x:xs,y:ys,keep_prob:1})
                train_writer.add_summary(train_result, i)
#else:
#    with tf.Session() as sess:
#        sess.run(init)
#        saver.restore(sess,'/home/cooper/PycharmProjects/GPU1/MCM/save_sess')
#        out=sess.run(big_loss,feed_dict={x:xs[0].reshape(1,68),y:ys[0].reshape(1,432),keep_prob:1})
#        print ys[0]
#        print out
#        out_return=pca_y.inverse_transform(out)
#        out_real=pca_y.inverse_transform(ys[0].reshape(1,49))
#        print out_return
#        print"##########################################################################################################"
#        print out_real
#        for i in range(68):
#            print out_real[i]#,out_return[i]




