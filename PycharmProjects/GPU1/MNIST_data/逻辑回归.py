"""     try add layer(100)
        but some unexpect bug happened
        gradientDescent exploding
"""
import tensorflow as tf
import numpy as np
import tensorflow.examples.tutorials.mnist.input_data as input_data
mnist=input_data.read_data_sets('MNIST_data',one_hot=True)
train_epochs=50
batch_szie=128
display=5
learning_rate=0.01
x=tf.placeholder("float",shape=[None,784])
y_=tf.placeholder("float",shape=[None,10])
W=tf.Variable(tf.random_normal([784,10],1.0,1.0))
b=tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)
cost = -tf.reduce_mean((y_ * tf.log(y)),reduction_indices=1)
train = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
    with tf.device('/gpu:0'):
        sess.run(tf.global_variables_initializer())
        for epoch in range(train_epochs):
            avg_cost=0
            num_batch=int(mnist.train.num_examples/batch_szie)
            for i in range(num_batch):
                batch_xs,batch_ys=mnist.train.next_batch(batch_szie)
                sess.run(train,feed_dict={x:batch_xs,y_:batch_ys})
                feeds={x:batch_xs,y:batch_ys}
                avg_cost+=sess.run(cost,feed_dict=feeds)/num_batch
            if epoch%display==0:
                feed_train={x:batch_xs,y:batch_ys}
                feed_test={x:mnist.test.images,y:mnist.test.labels}
                train_acc=sess.run(accuracy,feed_dict=feed_train)
                test_acc=sess.run(accuracy,feed_dict=feed_test)
                print "Epoch= ",epoch," avg_cost= ",avg_cost," train_acc= ",train_acc," test= ",test_acc



