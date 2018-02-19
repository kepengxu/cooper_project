import tensorflow as tf
import numpy as np
#import matplotlib.pyplot as plt
import tensorflow.examples.tutorials.mnist.input_data as input_data
mnist=input_data.read_data_sets('MNIST_data',one_hot=True)

#loading data mnist
in_put=784
n_classes=10
stddev=0.1
x=tf.placeholder(tf.float32,[None,in_put])
y=tf.placeholder(tf.float32,[None,n_classes])
weights={
    'wc1':tf.Variable(tf.random_normal([3,3,1,64],stddev=stddev)),
    'wc2':tf.Variable(tf.random_normal([3,3,64,128],stddev=stddev)),
    'wd1':tf.Variable(tf.random_normal([7*7*128,1024],stddev=stddev)),
    'wd2':tf.Variable(tf.random_normal([1024,10],stddev=stddev))
}
biases={
    'bc1':tf.Variable(tf.zeros([64])),
    'bc2':tf.Variable(tf.zeros([128])),
    'bd1':tf.Variable(tf.zeros([1024])),
    'bd2':tf.Variable(tf.zeros([n_classes]))
}
def conv(_x,_w,_b,_keepratio):
    in_put_r=tf.reshape(_x,[-1,28,28,1])#[batch_sizes,height,weight,chanels]
    conv1=tf.nn.conv2d(in_put_r,_w['wc1'],strides=[1,1,1,1],padding="SAME")#strides=[batch_size_step,h_step,w_step,chanels_step]
    conv1=tf.nn.relu(tf.nn.bias_add(conv1,_b['bc1']))
    pool1=tf.nn.max_pool(conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")#pool=[batch_size,h,w,chanels]
    pool_dr1=tf.nn.dropout(pool1,_keepratio)
    conv2=tf.nn.conv2d(pool_dr1,_w['wc2'],strides=[1,1,1,1],padding="SAME")
    conv2=tf.nn.relu(tf.nn.bias_add(conv2,_b['bc2']))
    pool2=tf.nn.max_pool(conv2,ksize=[1,1,1,1],strides=[1,2,2,1],padding="SAME")
    pool_dr2=tf.nn.dropout(pool2,_keepratio)
    _dense1=tf.reshape(pool_dr2,[-1,_w['wd1'].get_shape().as_list()[0]])
    ful_1=tf.nn.relu(tf.add(tf.matmul(_dense1,_w['wd1']),_b['bd1']))
    ful_dr1=tf.nn.dropout(ful_1,_keepratio)
    _out=tf.add(tf.matmul(ful_dr1,_w['wd2']),_b['bd2'])
    out = {
        'input_r': in_put_r, 'conv1': conv1, 'pool1': pool1,
        'pool_dr1': pool_dr1, 'conv2': conv2, 'pool2': pool_dr2,
        '_dense1': _dense1, '_fc1': ful_1, 'fc_dr1': ful_dr1, '_out': _out
    }
    return out
print "PRE OK"
keepratio = tf.placeholder(tf.float32)
pre=conv(x,weights,biases,keepratio)['_out']
print pre.shape
cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pre,labels=y))
optm=tf.train.AdamOptimizer(0.01).minimize(cost)
_corr=tf.equal(tf.argmax(pre,1),tf.argmax(y,1))
accr=tf.reduce_mean(tf.cast(_corr,tf.float32))
init=tf.global_variables_initializer()
saver=tf.train.Saver()
train_epochs=20
batch_size=64
display_step=1
#with tf.device('/gpu:0'):
with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:

#with tf.Session() as sess:
    sess.run(init)


    for epoch in range(train_epochs):
        avg_cost=0.0
        total_batch=int(mnist.train.num_examples/batch_size)
        for i in range(total_batch):
            batch_xs,batch_ys=mnist.train.next_batch(batch_size)
            sess.run(optm,feed_dict={x:batch_xs,y:batch_ys,keepratio:0.8})
            avg_cost+=sess.run(cost,feed_dict={x:batch_xs,y:batch_ys,keepratio:1.})/total_batch
        if (epoch+1)%display_step==0:
            trainacc=sess.run(accr,feed_dict={x:batch_xs,y:batch_ys,keepratio:1.0})
            print epoch
            print"Epoch:%03d/%03d cost   :%.9f   train_accuracy :%.3f"%(epoch,train_epochs,avg_cost,trainacc)
            feedtest={x:mnist.test.images,y:mnist.test.labels,keepratio:1}
            testacc=sess.run(accr,feed_dict=feedtest)
            print"test _accuracy",testacc



