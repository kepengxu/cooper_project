import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os as os
import tensorflow.examples.tutorials.mnist.input_data as input_data
mnist=input_data.read_data_sets('MNIST_data',one_hot=True)
n_input=784
n_hidden_1=254
n_hidden_2=64
n_classes=10
learningrate=0.005
x=tf.placeholder('float',[None,n_input])
y=tf.placeholder('float',[None,n_classes])
stddev=0.1
weights={
    'w1': tf.Variable(tf.random_normal([n_input,n_hidden_1],stddev=stddev)),
    'w2': tf.Variable(tf.random_normal([n_hidden_1,n_hidden_2],stddev=stddev)),
    'out': tf.Variable(tf.random_normal([n_hidden_2,n_classes],stddev=stddev))
}
biases={
    'b1': tf.Variable(tf.zeros([n_hidden_1])),
    'b2': tf.Variable(tf.zeros([n_hidden_2])),
    'out': tf.Variable(tf.zeros([n_classes]))
}
saver=tf.train.Saver()
print "NETWORK GO"
def multilayer_perceptron(_x,_weights, _biases):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(_x,_weights['w1']),_biases['b1']))
    print "layer_1",layer_1.shape,"_x",_x.shape
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1,_weights['w2']),_biases['b2']))
    print "layer_2", layer_2.shape
    return (tf.matmul(layer_2,_weights['out'])+_biases['out'])


pre = multilayer_perceptron(x,weights,biases)


cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pre,labels=y))
optm =tf.train.GradientDescentOptimizer(learningrate).minimize(cost)
corr = tf.equal(tf.argmax(pre,1),tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(corr,"float"))
init = tf.global_variables_initializer()
print "GO"
training_epochs=20
batch_size=128
display_step=5
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(training_epochs):
        avg_cost=0.0
        total_batch=int(mnist.train.num_examples/batch_size)
        for i in range(total_batch):
            batch_xs,batch_ys=mnist.train.next_batch(batch_size)
            feeds={x:batch_xs,y:batch_ys}
            sess.run(optm,feed_dict=feeds)
            avg_cost+=sess.run(cost,feed_dict=feeds)
            avg_cost=avg_cost/total_batch
        if(epoch+1)%display_step==0:
            print "epoch:%03d/%03d cost :%.9f"%(epoch,training_epochs,avg_cost)
            feeds={x:batch_xs,y:batch_ys}
            train_acc=sess.run(accuracy,feed_dict=feeds)
            print " Train accuracy: %.3f"%train_acc
            feeds={x:mnist.test.images,y:mnist.test.labels}
            test_acc=sess.run(accuracy,feed_dict=feeds)
            print "Test accuracy: %.3f"%test_acc