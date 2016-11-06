import tensorflow as tf
import numpy as np
from pq import PriorityQueue
from math import *

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

def global_inhibit(overlap_scores, out_size, max_active):
    # Set of outputs to keep
    keep = PriorityQueue()

    for i in range(out_size):
        score = overlap_scores[0][i]

        if keep.empty() or score > keep.peek():
            # We want this score to be kept

            if keep.size() < max_active:
                keep.update(i, score)
            else:
                # Replace the lowest score in keep with this one
                keep.pushpop(i, score)

    # Set all indices not kept to 0
    keep_indicies = {i for _, _, i in keep.heap}

    for i in range(out_size):
        if i not in keep_indicies:
            overlap_scores[0][i] = 0

    return overlap_scores

# Defines the condition to conert permanence into connections
p_cond = tf.greater(p, 0)

# Update the connections based on permenance
update_connections = tf.assign(c, tf.select(p_cond, tf.ones(weight_dim), tf.zeros(weight_dim)))

# Compute the activation before inhibition applied
activation = tf.matmul(x, c)

# TODO: Output vector
y = tf.while_loop(lambda a: tf_count(a, 0) < output_zeros, lambda a: tf.sub(a, tf.ones(1)), [activation])

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
        result = sess.run([activation], feed_dict={x: [[0,0,0,0,1,0,0,0,1,0]]})
        print(global_inhibit(result, 10, 2))
