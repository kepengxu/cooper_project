import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.examples.tutorials.mnist.input_data as input_data
mnist=input_data.read_data_sets('MNIST_data',one_hot=True)
train_epochs=50
x_labe=range(50)
test_arr=np.zeros(50)
batch_szie=128
display=5
learning_rate=0.01
x=tf.placeholder("float",shape=[None,784])
y=tf.placeholder("float",shape=[None,10])
W=tf.Variable(tf.random_normal([784,10],1.0,1.0))
b=tf.Variable(tf.zeros([10]))
actv= tf.nn.softmax(tf.matmul(x, W) + b)
cost = -tf.reduce_mean((y * tf.log(actv)),reduction_indices=1)
train = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(actv,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(train_epochs):
        avg_cost=0
        num_batch=int(mnist.train.num_examples/batch_szie)
        for i in range(num_batch):
            batch_xs,batch_ys=mnist.train.next_batch(batch_szie)
            sess.run(train,feed_dict={x:batch_xs,y:batch_ys})
            feeds={x:batch_xs,y:batch_ys}
            avg_cost+=sess.run(cost,feed_dict=feeds)/num_batch

        feed_train={x:batch_xs,y:batch_ys}
        feed_test={x:mnist.test.images,y:mnist.test.labels}
        train_acc=sess.run(accuracy,feed_dict=feed_train)
        test_acc=sess.run(accuracy,feed_dict=feed_test)
        test_arr[epoch] = test_acc
        print "Epoch= ",epoch," train_acc= ",train_acc," test= ",test_acc
    plt.scatter(x_labe,test_arr)
    plt.show()
