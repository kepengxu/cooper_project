import tensorflow as tf
import datetime
import numpy as np
#import matplotlib.pyplot as plt
import tensorflow.examples.tutorials.mnist.input_data as input_data
mnist=input_data.read_data_sets('MNIST_data',one_hot=True)

#loading data mnist
in_put=784
n_classes=10
stddev=0.1
def batch_norm_con(x,n_out,phase_train):
    '''
    batch_normalization on convolutional maps



    :param x:
    :param n_out:
    :param phase_train:
    :return:

    '''
    beta=tf.Variable(tf.zeros([n_out]),name='beta')
    gamma=tf.Variable(tf.ones([n_out]),name='gamma')
    batch_mean,batch_var=tf.nn.moments(x,[0,1,2],name='moments')
    ema=tf.train.ExponentialMovingAverage(decay=0.5)
    def mean_var_with_update():
        ema_apply_op=ema.apply([batch_mean,batch_var])
        with tf.control_dependencies([ema_apply_op]):
            return tf.identity(batch_mean),tf.identity(batch_var)
    mean,var = tf.cond(phase_train,
                        mean_var_with_update,
                        lambda: (ema.average(batch_mean), ema.average(batch_var)))
    normed=tf.nn.batch_normalization(x,mean,var,beta,gamma,1e-4)
    return normed
print "batch_normal is ok"


with tf.name_scope('inputs'):
    x=tf.placeholder(tf.float32,[None,in_put],name='input')
    y=tf.placeholder(tf.float32,[None,n_classes],name='output')
phase_train=tf.placeholder(tf.bool,name='phase_train')
def conv(_x,_keepratio,phase_train):
    with tf.name_scope('conv__1_layer'):
        with tf.name_scope('weights'):
            weights1=tf.Variable(tf.random_normal([3,3,1,32],stddev=stddev,name='wc1'))
            tf.summary.histogram('conv1/weight',weights1)
        in_put_r=tf.reshape(_x,[-1,28,28,1],name='reshape_input')#[batch_sizes,height,weight,chanels]
        conv1=tf.nn.conv2d(in_put_r,weights1,strides=[1,1,1,1],padding="SAME",name='conv1')#strides=[batch_size_step,h_step,w_step,chanels_step]
        conv1bn=batch_norm_con(conv1,32,phase_train=phase_train)
        conv1=tf.nn.relu(conv1bn)
        pool1=tf.nn.max_pool(conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME",name='max_pool_1')#pool_window_size=[batch_size,h,w,chanels]
        #print pool1.size()
        pool_dr1=tf.nn.dropout(pool1,_keepratio,name='dropout1')
    with tf.name_scope('conv__2_layer'):
        with tf.name_scope('weight'):
            weights2=tf.Variable(tf.random_normal([5,5,32,64],stddev=stddev,name='wc2')),
            tf.summary.histogram('conv2/weight',weights2)
        conv2=tf.nn.conv2d(pool_dr1,weights2,strides=[1,1,1,1],padding="SAME",name='conv2')
        conv2bn = batch_norm_con(conv2, 64,phase_train=phase_train)
        conv2=tf.nn.relu(conv2bn)
        pool2=tf.nn.max_pool(conv2,ksize=[1,1,1,1],strides=[1,2,2,1],padding="SAME",name='max_pool2')
        #print pool2.size
        pool_dr2=tf.nn.dropout(pool2,_keepratio,name='dropout2')
    with tf.name_scope('conv__3_layer'):
        with tf.name_scope('weight'):
            weights3=tf.Variable(tf.random_normal([3,3,64,128],stddev=stddev,name='wc3'))
            tf.summary.histogram('conv3/weight', weights3)
        conv3 = tf.nn.conv2d(pool_dr2, weights3, strides=[1, 1, 1, 1], padding="SAME", name='conv3')
        conv3bn = batch_norm_con(conv3, 128,phase_train=phase_train)
        conv3 = tf.nn.relu(conv3bn)
        pool3 = tf.nn.max_pool(conv3, ksize=[1, 1, 1, 1], strides=[1, 2, 2, 1], padding="SAME", name='max_pool3')
        #print pool3.size
        pool_dr3 = tf.nn.dropout(pool3, _keepratio, name='dropout3')
    with tf.name_scope('full_connected_layer'):
        with tf.name_scope('weight'):
            weights_full=tf.Variable(tf.random_normal([4*4*128,1024],stddev=stddev,name='wd1'))
            tf.summary.histogram('full_connected/weight', weights_full)
        with tf.name_scope('bias'):
            bias_f=tf.Variable(tf.zeros([1024],name='bd1'))
            tf.summary.histogram('full_connected/bias', bias_f)
        _dense1=tf.reshape(pool_dr3,[-1,weights_full.get_shape().as_list()[0]])
        print _w['wd1'].get_shape().as_list()[0]
        print"***"
        print 4*4*128
        ful_1=tf.nn.relu(tf.add(tf.matmul(_dense1,weights_full), bias_f),name='full_con1')
        ful_dr1=tf.nn.dropout(ful_1,_keepratio)
        with tf.name_scope('weight_class'):
            weight_f2=tf.Variable(tf.random_normal([1024,10],stddev=stddev,name='wd2'))
            bias_f2=tf.Variable(tf.zeros([n_classes],name='bd2'))
        _out=tf.add(tf.matmul(ful_dr1,weight_f2),bias_f2,name='_out')
        out = {
            '_out': _out
        }
    return out
print "PRE OK"
keepratio = tf.placeholder(tf.float32)
pre=conv(x,keepratio,phase_train)['_out']
print pre.shape
with tf.name_scope('cost'):
    cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pre,labels=y))
    tf.summary.scalar('cost',cost)
optm=tf.train.AdamOptimizer(0.001).minimize(cost,name='train')
_corr=tf.equal(tf.argmax(pre,1),tf.argmax(y,1))
with tf.name_scope('accr'):
    accr=tf.reduce_mean(tf.cast(_corr,tf.float32),name='accr')
init=tf.global_variables_initializer()
saver=tf.train.Saver()
train_epochs=50
batch_size=16
display_step=2
config=tf.ConfigProto(device_count={"CPU":3},
                      inter_op_parallelism_threads=3,
                      intra_op_parallelism_threads=3,
                      allow_soft_placement=True
                      #log_device_placement=True
)
j=0
with tf.Session() as sess:
    sess.run(init)
#    saver.restore(sess,'model/modelCNN_graph.ckpt')
#    feedtest = {x: mnist.test.images, y: mnist.test.labels, keepratio: 1, phase_train: False}
#    testacc = sess.run(accr, feed_dict=feedtest)
#    print"test _accuracy", testacc
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("logsss/", sess.graph)
    sess.run(init)
    ti = datetime.datetime.now()
    for epoch in range(train_epochs):
        avg_cost=0.0
        #total_batch=int(mnist.train.num_examples/batch_size)
        total_batch=20
        for i in range(total_batch):
            j=j+1
            batch_xs,batch_ys=mnist.train.next_batch(batch_size)
            sess.run(optm,feed_dict={x:batch_xs,y:batch_ys,keepratio:0.8,phase_train:True})
            avg_cost+=sess.run(cost,feed_dict={x:batch_xs,y:batch_ys,keepratio:1.,phase_train:False})/total_batch
            if j%5==0:
                result = sess.run(merged,
                                  feed_dict={x: batch_xs, y: batch_ys, keepratio: 1.0, phase_train: False})
                writer.add_summary(result, j)
        if (epoch+1)%display_step==0:
            daita = datetime.datetime.now() - ti
            print daita.total_seconds()
            trainacc=sess.run(accr,feed_dict={x:batch_xs,y:batch_ys,keepratio:1.0,phase_train:False})
            print epoch
            print"Epoch:%03d/%03d cost   :%.9f   train_accuracy :%.3f"%(epoch,train_epochs,avg_cost,trainacc)

#        saver.save(sess,"model/modelCNN_graph.ckpt")



