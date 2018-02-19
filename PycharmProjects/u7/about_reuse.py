import tensorflow as tf
with tf.variable_scope('v_scope',reuse=True) as scope:
    weights2=tf.get_variable('weights')
    print weights2.name