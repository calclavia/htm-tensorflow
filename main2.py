import tensorflow as tf
import numpy as np
from pq import PriorityQueue
from math import *

inhibit_module = tf.load_op_library('./inhibit.so')

epochs = 1
dim = (10, 10)
sparsity = 0.2

output_zeros = int(dim[1] * (1 - sparsity))

# Some input vector
x = tf.placeholder(tf.float32)

weight_dim = [10, 10]

# Permanence of connections between neurons
p = tf.Variable(tf.random_uniform(weight_dim, -1, 1), name='Permanence')

# TODO: Synapse matrix for what can possibly connect

# Connection matrix
c = tf.Variable(tf.zeros(weight_dim), name='Connections')

"""
Operations
"""
def tf_count(t, val):
    elements_equal_to_value = tf.equal(t, val)
    as_ints = tf.cast(elements_equal_to_value, tf.int32)
    count = tf.reduce_sum(as_ints)
    return count

# Defines the condition to conert permanence into connections
p_cond = tf.greater(p, 0)

# Update the connections based on permenance
update_connections = tf.assign(c, tf.select(p_cond, tf.ones(weight_dim), tf.zeros(weight_dim)))

# Compute the activation before inhibition applied
activation = tf.matmul(x, c)

# Output vector
y = inhibit_module.inhibit(activation)

init_op = tf.initialize_all_variables()

# To run the matmul op we call the session 'run()' method, passing 'product'
# which represents the output of the matmul op.  This indicates to the call
# that we want to get the output of the matmul op back.
#
# All inputs needed by the op are run automatically by the session.  They
# typically are run in parallel.
#
# The call 'run(product)' thus causes the execution of three ops in the
# graph: the two constants and matmul.
#
# The output of the op is returned in 'result' as a numpy `ndarray` object.
with tf.Session() as sess:
    # Run the 'init' op
    sess.run(init_op)
    sess.run(update_connections)

    for _ in range(1):
        result, = sess.run([y], feed_dict={x: [[0,0,0,0,1,0,0,0,1,0]]})
        print(result)
