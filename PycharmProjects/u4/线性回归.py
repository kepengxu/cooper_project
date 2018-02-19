import tensorflow as tf
import numpy as np
num_points=1000
vectors_set=[]
for i in range(num_points):
    x1=np.random.normal(0.0,0.55,)
    y1=x1*0.1+0.3+np.random.normal(0.0,0.03)
    vectors_set.append([x1,y1])
x_data=[v[0] for v in vectors_set]
y_data=[v[1] for v in vectors_set]
w=tf.Variable(tf.random_uniform([1],1.0,-1.0),name='weight')
#tf.histogram_fixed_width('weight',w)

b=tf.Variable(tf.zeros([1]),name='bias')
#tf.histogram_fixed_width('biases',b)
y=x_data*w+b
loss=tf.reduce_mean(tf.square(y-y_data),name='loss')
tf.summary.scalar('loss',loss)
train=tf.train.AdamOptimizer(0.002).minimize(loss,name='train')
with tf.Session() as sess:
    writer = tf.summary.FileWriter("tbb/", sess.graph)
    init=tf.global_variables_initializer()
    sess.run(init)
    print "w=",sess.run(w),"  b=",sess.run(b)," loss=",sess.run(loss)
    for i in range(10000):
        sess.run(train)
        if loss.eval()<0.0001:
            break;
        if i%100==0:
            print i,"           w=",sess.run(w),"  b=",sess.run(b)," loss=",sess.run(loss)