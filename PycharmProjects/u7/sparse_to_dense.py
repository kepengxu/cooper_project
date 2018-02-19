import tensorflow as tf
batch_size=7
label=tf.expand_dims(tf.constant([0,2,5,6,3,9,8]),1)
index=tf.expand_dims(tf.range(7),1)
concat=tf.concat([index,label],1)
one_hot=tf.sparse_to_dense(concat,[batch_size,10],1.0,0.0)
print one_hot
with tf.Session() as sess:
    print one_hot