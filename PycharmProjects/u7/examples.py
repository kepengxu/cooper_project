import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.examples.tutorials.mnist.input_data as input_data
mnist=input_data.read_data_sets('MNIST_data',one_hot=True)
learning_rate = 0.1
batch_size = 64

n_input = 28
n_steps = 28
n_hidden = 128
n_classes = 10
x = tf.placeholder(tf.float32, [None, n_steps, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])


# In[5]:
sess=tf.Session()
#def RNN(x, weight, biases):
#    # x shape: (batch_size, n_steps, n_input)
#    # desired shape: list of n_steps with element shape (batch_size, n_input)
#    x = tf.transpose(x, [1, 0, 2])
#    x = tf.reshape(x, [-1, n_input])
#    x = tf.split(0, n_steps, x)
#    outputs = list()
#    lstm = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)
#    state = (tf.zeros([n_steps, n_hidden]),) * 2
#    sess.run(state)
#    with tf.variable_scope("myrnn2") as scope:
#        for i in range(n_steps - 1):
#            if i > 0:
#                scope.reuse_variables()
#            output, state = lstm(x[i], state)
#            outputs.append(output)
#    final = tf.matmul(outputs[-1], weight) + biases
#    return final
# In[6]:
def RNN(x, n_steps, n_input, n_hidden, n_classes):
    # Parameters:
    # Input gate: input, previous output, and bias
    ix = tf.Variable(tf.truncated_normal([n_input, n_hidden], -0.1, 0.1))
    im = tf.Variable(tf.truncated_normal([n_hidden, n_hidden], -0.1, 0.1))
    ib = tf.Variable(tf.zeros([1, n_hidden]))
    # Forget gate: input, previous output, and bias
    fx = tf.Variable(tf.truncated_normal([n_input, n_hidden], -0.1, 0.1))
    fm = tf.Variable(tf.truncated_normal([n_hidden, n_hidden], -0.1, 0.1))
    fb = tf.Variable(tf.zeros([1, n_hidden]))
    # Memory cell: input, state, and bias
    cx = tf.Variable(tf.truncated_normal([n_input, n_hidden], -0.1, 0.1))
    cm = tf.Variable(tf.truncated_normal([n_hidden, n_hidden], -0.1, 0.1))
    cb = tf.Variable(tf.zeros([1, n_hidden]))
    # Output gate: input, previous output, and bias
    ox = tf.Variable(tf.truncated_normal([n_input, n_hidden], -0.1, 0.1))
    om = tf.Variable(tf.truncated_normal([n_hidden, n_hidden], -0.1, 0.1))
    ob = tf.Variable(tf.zeros([1, n_hidden]))
    # Classifier weights and biases
    w = tf.Variable(tf.truncated_normal([n_hidden, n_classes]))
    b = tf.Variable(tf.zeros([n_classes]))

    # Definition of the cell computation
    def lstm_cell(i, o, state):
        input_gate = tf.sigmoid(tf.matmul(i, ix) + tf.matmul(o, im) + ib)
        forget_gate = tf.sigmoid(tf.matmul(i, fx) + tf.matmul(o, fm) + fb)
        update = tf.tanh(tf.matmul(i, cx) + tf.matmul(o, cm) + cb)
        state = forget_gate * state + input_gate * update
        output_gate = tf.sigmoid(tf.matmul(i, ox) + tf.matmul(o, om) + ob)
        return output_gate * tf.tanh(state), state

    # Unrolled LSTM loop
    outputs = list()
    state = tf.Variable(tf.zeros([batch_size, n_hidden]))
    output = tf.Variable(tf.zeros([batch_size, n_hidden]))

    # x shape: (batch_size, n_steps, n_input)
    # desired shape: list of n_steps with element shape (batch_size, n_input)
    x = tf.transpose(x, [1, 0, 2])
    x = tf.reshape(x, [-1, n_input])
    x = tf.split(x, n_steps, 0)
    for i in x:
        output, state = lstm_cell(i, output, state)
        outputs.append(output)
    logits = tf.nn.softmax(tf.matmul(outputs[-1], w) + b)
    return logits


# In[7]:

pred = RNN(x, n_steps, n_input, n_hidden, n_classes)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

# In[8]:

# Launch the graph
sess.run(init)
for step in range(20000):
    batch_x, batch_y = mnist.train.next_batch(batch_size)
    batch_x = batch_x.reshape((batch_size, n_steps, n_input))
    sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})

    if step % 50 == 0:
        acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
        loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
        print "Iter " + str(step) + ", Minibatch Loss= " + "{:.6f}".format(
            loss) + ", Training Accuracy= " + "{:.5f}".format(acc)
print "Optimization Finished!"

# In[9]:

# Calculate accuracy for 128 mnist test images
test_len = batch_size
test_data = mnist.test.images[:test_len].reshape((-1, n_steps, n_input))
test_label = mnist.test.labels[:test_len]
print "Testing Accuracy:", sess.run(accuracy, feed_dict={x: test_data, y: test_label})
