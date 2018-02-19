import tensorflow as tf
import numpy as np
NUM_CLASSES=10
labels=[0,8,2,3,4,5]
batch_sizes=tf.size(labels)
labels=tf.expand_dims(labels,1)
print "hh",labels.shape
indices=tf.expand_dims(tf.range(0,batch_sizes,),1)
#indices=tf.constant([[5],[4],[3],[2],[1],[0]])
concated=tf.concat([indices,labels],1)
onehot_labels=tf.sparse_to_dense(concated,[batch_sizes,NUM_CLASSES],1.0,0.0)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print labels.shape
    print "ONE_HOT_SHAPE"
    print sess.run(onehot_labels)
    print "label_SHAPE"
    print sess.run(labels)
    print "indices"
    print sess.run(indices)
    print "concated"
    print sess.run(concated)