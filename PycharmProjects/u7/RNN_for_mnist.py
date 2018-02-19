import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.examples.tutorials.mnist.input_data as input_data
from tensorflow.contrib import rnn
mnist=input_data.read_data_sets('MNIST_data',one_hot=True)
print tf.__version__
n_classes=10
dim_input=28
dim_hidden=128
n_step=28
stddev=0.1
batch_size=1
weights={
    'hidden':tf.Variable(tf.random_normal([dim_input,dim_hidden],stddev=stddev)),
    'out':tf.Variable(tf.random_normal([dim_hidden,n_classes],stddev=stddev))
}
biases={
    'hidden':tf.Variable(tf.zeros([dim_hidden])),
    'out':tf.Variable(tf.zeros([n_classes]))
}

def _RNN(_x,_w,_b,_nsteps):
    _x=tf.reshape(_x,[-1,dim_input])
    _x=tf.matmul(_x,_w['hidden'])+_b['hidden']
    _x=tf.reshape(_x,[-1,_nsteps,dim_hidden])
    lstm_cell=tf.nn.rnn_cell.BasicLSTMCell(dim_hidden,forget_bias=1.0,state_is_tuple=True)
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
            train_acc=sess.run(accr,feed_dict={x:batch_xs,y:batch_ys})
            bx=mnist.test.images
            bx.tf.reshape(bx,[n_step,dim_input])
            feed_t={x:bx,y:mnist.test.labels}
            test_acc=sess.run(accr,feed_dice=feed_t)
            print " %03d   /%03d   cost  %0.9f    train_acc   %0.3f    test_acc    %.3f"%(epoch,total_epoches,avg_cost,train_acc,test_acc)




#print _x.shape
##    _x=tf.transpose(_x,[1,0,2])#[step,batch,dim_input]
 #   print _x.shape
 #   _x=tf.reshape(_x,[-1,dim_input])  #[batch_size*step,dim_input]
 #   _h=tf.matmul(_x,_w['hidden'])+_b['hidden']
 #   _hsplit=tf.split(_h,_nsteps,0)
 #   with tf.variable_scope(_name) as scope:
 #       scope.reuse_variables()
 #       lstm_cell=tf.nn.rnn_cell.BasicLSTMCell(dim_hidden,forget_bias=1.0)
 #       _LSTM_O,_LSTM_S=tf.nn.dynamic_rnn(lstm_cell,_hsplit,dtype=tf.float32)
#        print _LSTM_O.shape
#        print ">>>>>>>>>",_w['out']
 #       _O=tf.matmul(_LSTM_O[-1],_w['out'])+_b['out']
 #   return {
 #       'x':_x,'h':_h,'hsplit':_hsplit,'LSTM_O':_LSTM_O,'LSTM_S':_LSTM_S,'o':_O}