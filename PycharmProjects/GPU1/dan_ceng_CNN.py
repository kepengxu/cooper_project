"""
this code have loding moxing
"""
import tensorflow as tf
import datetime
import numpy as np
#import matplotlib.pyplot as plt
import tensorflow.examples.tutorials.mnist.input_data as input_data
mnist=input_data.read_data_sets('MNIST_data',one_hot=True)
#print help(tf.shape)
n_input=784
n_output=10
stddev=0.1
weights={
    'wc1':tf.Variable(tf.random_normal([3,3,1,128],stddev=stddev)),
    'wc2':tf.Variable(tf.random_normal([3,3,64,128],stddev=stddev)),
    'wd1':tf.Variable(tf.random_normal([14*14*128,1024],stddev=stddev)),
    'wd2':tf.Variable(tf.random_normal([1024,10],stddev=stddev))
}
biases={
    'bc1':tf.Variable(tf.zeros([128])),
    'bc2':tf.Variable(tf.zeros([128])),
    'bd1':tf.Variable(tf.zeros([1024])),
    'bd2':tf.Variable(tf.zeros([n_output]))
}
def conv(_input,_w,_b,_keepratio):
    #input
    _input_r=tf.reshape(_input,shape=[-1,28,28,1])
    conv1=tf.nn.conv2d(_input_r,weights['wc1'],strides=[1,1,1,1],padding="SAME")
    conv1=tf.nn.relu(tf.nn.bias_add(conv1,biases['bc1']))
    pool1=tf.nn.max_pool(conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")
    pool_dr1=tf.nn.dropout(pool1,_keepratio)
#    conv2=tf.nn.conv2d(pool_dr1,weights['wc2'],strides=[1,1,1,1],padding="SAME")
#    conv2=tf.nn.relu(tf.nn.bias_add(conv2,biases['bc2']))
#    pool2=tf.nn.max_pool(conv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")
#    pool_dr2=tf.nn.dropout(pool2,_keepratio)
 #   print "pool_dr2.shape",pool_dr2.shape
#    print "_w['wd1'].get_shape().as_list()[0]",_w['wd1'].get_shape().as_list()[0]
    _dense1=tf.reshape(pool_dr1,[-1,_w['wd1'].get_shape().as_list()[0]])
    print _dense1.shape
    _fc1=tf.nn.relu(tf.add(tf.matmul(_dense1,_w['wd1']),_b['bd1']))
    fc_dr1=tf.nn.dropout(_fc1,_keepratio)
    _out=tf.add(tf.matmul(fc_dr1,_w['wd2']),_b['bd2'])
    out={
#        'input_r':_input_r,'conv1':conv1,'pool1':pool1,
#        'pool_dr1':pool_dr1,'conv2':conv2,'pool2':pool_dr2,
#        '_dense1':_dense1,'_fc1':_fc1,'fc_dr1':fc_dr1,
        '_out':_out
    }
    return out
print "CNN OK"
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_output])
keepratio = tf.placeholder(tf.float32)
#with tf.device('/gpu:0'):
pre = conv(x, weights, biases, keepratio)['_out']
print "OOOOO"
with tf.name_scope('cost'):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pre, labels=y))
optm = tf.train.AdamOptimizer(0.001).minimize(cost)
_corr = tf.equal(tf.argmax(pre, 1), tf.argmax(y, 1))
accr = tf.reduce_mean(tf.cast(_corr, tf.float32))
init = tf.global_variables_initializer()
#saver=tf.train.Saver()
print "GRAPH OK"
train_epochs=64
batch_size=128
display_step=1
#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
config=tf.ConfigProto(
                      #device_count={"CPU":0},
                      inter_op_parallelism_threads=3,
                      intra_op_parallelism_threads=3,
                      allow_soft_placement=True
                      #log_device_placement=True
)
configg = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=configg) as sess:
    sess.run(init)
    ti = datetime.datetime.now()
    for epoch in range(train_epochs):
        avg_cost=0.0
        total_batch=int(mnist.train.num_examples/batch_size)
        #print total_batch
        #total_batch=200
        for i in range(
                total_batch):
            batch_xs,batch_ys=mnist.train.next_batch(batch_size)
            sess.run(optm,feed_dict={x:batch_xs,y:batch_ys,keepratio:0.8})
            avg_cost+=sess.run(cost,feed_dict={x:batch_xs,y:batch_ys,keepratio:1.})/total_batch
        if epoch%display_step==0:
#            with tf.device('/cpu:0'):
            daita = datetime.datetime.now() - ti
            print daita.total_seconds()
            print "epoch:%03d/%03d   cost   :%.9f"%(epoch,train_epochs,avg_cost)
            trainacc=sess.run(accr,feed_dict={x:batch_xs,y:batch_ys,keepratio:1.})
            print " Train accuracy: %.3f" % trainacc
            #feeds={x:mnist.test.images,y:mnist.test.labels,keepratio:1.}
            #test_acc=sess.run(accr,feed_dict=feeds)
            #print "Test accuracy: %.3f"%test_acc
    #saver.save(sess,"model/CNN_model_time"+str(epoch)+".ckpt")
    print "The Graph pram have save"
#with tf.Session() as sess:
#    saver.restore(sess,"model/CNN_model_time5.ckpt")
#    feeds = {x: mnist.test.images, y: mnist.test.labels, keepratio: 1.}
#    test_acc = sess.run(accr, feed_dict=feeds)
#    print "Test accuracy: %.3f"%test_acc