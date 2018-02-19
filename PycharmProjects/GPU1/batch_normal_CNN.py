import tensorflow as tf
import numpy as np
import datetime
import tensorflow.examples.tutorials.mnist.input_data as input_data
mnist=input_data.read_data_sets('MNIST_data',one_hot=True)


###initialized net structure
class convolution2d(object):

    def __init__(self,input,input_size,input_channels,output_channels,filter_size,NAME,activate='relu'):
        self.input=input
        self.row=input_size[0]
        self.col=input_size[1]
        self.input_channels=input_channels
        self.activate=activate
        wshape=[filter_size[0],filter_size[1],input_channels,output_channels]
        with tf.name_scope(NAME):
            w=tf.Variable(tf.random_normal(wshape,stddev=0.1),name='weight')
            b=tf.Variable(tf.zeros([output_channels]),name='bias')
            tf.summary.histogram(NAME+'weight',w)
            tf.summary.histogram(NAME+'bias',b)
        self.wc=w
        self.bc=b

    def output(self):
        shape4d=[-1,self.row,self.col,self.input_channels]
        batch_image2d=tf.reshape(self.input,shape4d)
        out=tf.add(tf.nn.conv2d(batch_image2d,self.wc,strides=[1,1,1,1],padding='SAME'),self.bc)
        if self.activate=='relu':
            self.output=tf.nn.relu(out)
        else:
            self.output=out
        return self.output

print "convolution is ok"
class maxpool2d(object):

    def __init__(self,input,ksize=None):
        self.input=input
        if ksize==None:
            ksize=[1,2,2,1]
        self.ksize=ksize



    def output(self):
        self.output=tf.nn.max_pool(self.input,ksize=self.ksize,strides=[1,2,2,1],padding="SAME")
        return self.output


print"maxpool2d is ok"
class fullconnected(object):

    def __init__(self,input,n_in,n_out,activate='relu'):
        self.input=input
        w=tf.Variable(tf.random_normal([n_in,n_out],mean=0.1,stddev=0.1))
        b=tf.Variable(tf.zeros([n_out]))
        self.wf=w
        self.bf=b
        self.activate=activate

    def output(self):
        l=tf.add(tf.matmul(self.input,self.wf),self.bf)
        if self.activate=='relu':
            self.fl=tf.nn.relu(l)
        else:
            self.fl=l
        return self.fl
print "fullconnected is ok"

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


class readoutlayer(object):
    def __init__(self,input,n_in,n_out):
        self.input=input
        w_o=tf.Variable(tf.random_normal([n_in,n_out],stddev=0.1))
        b_o=tf.Variable(tf.zeros([n_out]))
        self.w=w_o
        self.b=b_o
    def output(self):
        l=tf.add(tf.matmul(self.input,self.w),self.b)
        self.output=tf.nn.softmax(l)
        return self.output
def inference(x,y,keep_prob,phase_train):
    x=tf.reshape(x,[-1,28,28,1])
    #########################################conv1
    with tf.name_scope('conv_1'):
        conv1=convolution2d(x,[28,28],1,64,[5,5],activate='none',NAME='conv1')
        conv1bn=batch_norm_con(conv1.output(),64,phase_train=phase_train)
        conv1_out=tf.nn.relu(conv1bn)
        pool1=maxpool2d(conv1_out)
        pool1_out=pool1.output()


    ##########################################conv2
    with tf.name_scope('conv2'):
        conv2=convolution2d(pool1_out,(14,14),64,32,(3,3),activate='none',NAME='conv2')
        conv2bn=batch_norm_con(conv2.output(),32,phase_train=phase_train)
        conv2_out=tf.nn.relu(conv2bn)
        pool2=maxpool2d(conv2_out)
        pool2_out=pool2.output()


    #########################################conv3
    with tf.name_scope('conv3'):
        conv3=convolution2d(pool2_out,(7,7),32,128,(3,3),activate='none',NAME='conv3')
        conv3bn=batch_norm_con(conv3.output(),128,phase_train=phase_train)
        conv3_out=tf.nn.relu(conv3bn)

        pool3=maxpool2d(conv3_out)
        pool3_out=pool3.output()
        print pool3_out.shape

        pool3_flat=tf.reshape(pool3_out,[-1,128*4*4])

    ######################################fullconnected
    with tf.name_scope('fullconnected'):
        fc1=fullconnected(pool3_flat,128*4*4,1024)
        fc1_out=fc1.output()
        fc1_dropped=tf.nn.dropout(fc1_out,keep_prob=keep_prob)
    with tf.name_scope('soft_max'):
        w_o = tf.Variable(tf.random_normal([1024,10], stddev=0.1))
        b_o = tf.Variable(tf.zeros([10]))
        pred=tf.add(tf.matmul(fc1_dropped,w_o),b_o)
    #with tf.name_scope('cost'):
        #pred=readoutlayer(fc1_dropped,1024,10).output()
    #    pred=fc1_dropped
    return pred
print "inference is ok"
x=tf.placeholder(tf.float32,[None,784])
y=tf.placeholder(tf.float32,[None,10])
keep_prob=tf.placeholder(tf.float32)
phase_train=tf.placeholder(tf.bool,name='phase_train')
pre=inference(x,y,keep_prob,phase_train)
#cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pre), reduction_indices=[1]))
with tf.name_scope('cost'):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pre, labels=y))
    tf.summary.scalar('cost',cost)
optm=tf.train.AdamOptimizer(0.01).minimize(cost,name='train')
corr=tf.equal(tf.argmax(pre,1),tf.argmax(y,1))
with tf.name_scope('accuracy'):
    accuracy=tf.reduce_mean(tf.cast(corr,tf.float32),name='accuracy')
    tf.summary.scalar('accuracy',accuracy)
init=tf.global_variables_initializer()
total_epoch=100
batch_size=256
display_step=1
with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    merge=tf.summary.merge_all()
    writer = tf.summary.FileWriter("log/", sess.graph)
    sess.run(init)
    ti = datetime.datetime.now()

    for epoch in range(total_epoch):
        avgcost=0.0
        total_batch=int(mnist.train.num_examples/batch_size)
        #print "total batch",total_batch
        total_batch=20
        for i in range(total_batch):
            batch_xs,batch_ys=mnist.train.next_batch(batch_size)
            ####it is easyly to make bug
            feed_train={x:batch_xs,y:batch_ys,keep_prob:0.8,phase_train:True}
            feed_train_batch_cost={x:batch_xs,y:batch_ys,keep_prob:1.0,phase_train:False}

            sess.run(optm,feed_dict=feed_train)
            avgcost+=sess.run(cost,feed_dict=feed_train_batch_cost)
        if (epoch+1)%display_step==0:
            feed_train_batch_accuracy = {x: batch_xs, y: batch_ys, keep_prob: 1.0, phase_train: False}
            result=sess.run(merge,feed_dict=feed_train_batch_accuracy)
            writer.add_summary(result,epoch)
            daita=datetime.datetime.now()-ti
            print daita.total_seconds()
            train_accuracy=sess.run(accuracy,feed_dict=feed_train_batch_accuracy)
            print "epoch%03d/%03d  cost %.9f   train_accuracy  %.3f " % (epoch, total_epoch,avgcost, train_accuracy)



