import sys
import tensorflow as tf
print sys.getsizeof(tf.float32)
total=10000*28*28*128*64
print total
total/=1024
print total
total=total/1024
print total