import tensorflow as tf
import numpy as np
#import matplotlib.pyplot as plt
import datetime
import tensorflow.examples.tutorials.mnist.input_data as input_data
from tensorflow.contrib import rnn
mnist=input_data.read_data_sets('MNIST_data',one_hot=True)
print tf.__version__
n_classes=10
dim_input=28
dim_hidden1=128
dim_hidden2=256
n_step=28
stddev=0.1
batch_size=128
weights={
    'hidden1':tf.Variable(tf.random_normal([dim_input,dim_hidden1],stddev=stddev)),
    'hidden2':tf.Variable(tf.random_normal([dim_hidden1,dim_hidden2],stddev=stddev)),
    'out':tf.Variable(tf.random_normal([dim_hidden2,n_classes],stddev=stddev))
}
biases={
    'hidden1':tf.Variable(tf.zeros([dim_hidden1])),
    'hidden2':tf.Variable(tf.zeros([dim_hidden2])),
    'out':tf.Variable(tf.zeros([n_classes]))
}

def _RNN(_x,_w,_b,_nsteps):
    _x=tf.reshape(_x,[-1,dim_input])
    _x=tf.nn.relu(tf.matmul(_x,_w['hidden1'])+_b['hidden1'],name='layer1')
    _x = tf.nn.relu(tf.matmul(_x, _w['hidden2']) + _b['hidden2'], name='layer2')
    _x=tf.reshape(_x,[-1,_nsteps,dim_hidden2])
    lstm_cell=tf.nn.rnn_cell.BasicLSTMCell(dim_hidden2,forget_bias=1.0,state_is_tuple=True)
    init_state=lstm_cell.zero_state(batch_size,dtype=tf.float32)
    print ">>>>>",init_state
    _o,_s=tf.nn.dynamic_rnn(lstm_cell,_x,initial_state=init_state,time_major=False)
    print _s
    print _w['out'].shape
    return tf.matmul(_s[1],_w['out'])+_b['out']
print "RNN OK"
learning_rate=0.001
x=tf.placeholder(tf.float32,[None,n_step,dim_input])
y=tf.placeholder(tf.float32,[None,n_classes])
pre=_RNN(x,weights,biases,n_step)
cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pre,labels=y))
optm=tf.train.AdamOptimizer(learning_rate).minimize(cost)
accr=tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pre,1),tf.argmax(y,1)),tf.float32))
init=tf.global_variables_initializer()
print"NET OK"
total_epoches=100
dis_play=1
with tf.Session() as sess:
    ti=datetime.datetime.now()
    sess.run(init)
    for epoch in range(total_epoches):
        avg_cost=0.0
        total_batch=mnist.train.num_examples/batch_size
        for i in range(total_batch):
            batch_xs,batch_ys=mnist.train.next_batch(batch_size)
            batch_xs=batch_xs.reshape(batch_size,n_step,dim_input)
            sess.run(optm,feed_dict={x:batch_xs,y:batch_ys})
            avg_cost+=sess.run(cost,feed_dict={x:batch_xs,y:batch_ys})
        if (epoch+1)%5==0:
            daita=datetime.datetime.now()-ti
            print daita
            train_acc=sess.run(accr,feed_dict={x:batch_xs,y:batch_ys})
            print "Epoch:   %03d  cost :  %.9f   accuracy :  %.3f"%(epoch,avg_cost,train_acc)