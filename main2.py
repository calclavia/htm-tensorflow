import tensorflow as tf
import numpy as np
from pq import PriorityQueue
from math import *

inhibit_module = tf.load_op_library('./inhibit.so')

epochs = 10
dim = [10, 10]
sparsity = 0.2
learning_rate = 0.1

output_zeros = int(dim[1] * (1 - sparsity))

# Some input vector
x = tf.placeholder(tf.float32)

# Permanence of connections between neurons
p = tf.Variable(tf.random_uniform(dim, -1, 1), name='Permanence')

# TODO: Synapse matrix for what can possibly connect

# Connection matrix
c = tf.Variable(tf.zeros(dim), name='Connections')

"""
Operations
"""
# Defines the condition to conert permanence into connections
p_cond = tf.greater(p, 0)

# Update the connections based on permenance
update_connections = tf.assign(c, tf.select(p_cond, tf.ones(dim), tf.zeros(dim)))

# Compute the activation before inhibition applied
activation = tf.matmul(x, c)

# Output vector
y = inhibit_module.inhibit(activation)

"""
Weight update
"""
# Compute the delta permanence matrix
# Start with the connection matrix
# Every connection that was aligned with an input = 1. 0 otherwise.
input_rowed = tf.transpose(tf.matmul(tf.transpose(x), tf.ones_like(x)))
# Elementwise product between connection matrix and the input rows
# Rescale values to be between -1 and 1
alignment = (c * input_rowed - 0.5) * 2
# Only update weights to active outputs, so zero-out non-active output rows
# We do this by constructing a diagonal matrix that scales rows to zero
zero_out = tf.diag(y[0])
delta = learning_rate * tf.transpose(tf.matmul(tf.transpose(alignment), zero_out))
learn = tf.assign(p, tf.add(p, delta))

init_op = tf.initialize_all_variables()

def cluster():
    counts = [0 for _ in range(classes)]
    clusters = [np.mat(np.zeros((1, output_size))) for _ in range(classes)]
    for input, c in zip(images, labels):
        clusters[c] += layer.forward(input)
        counts[c] += 1

    for c in range(len(clusters)):
        clusters[c] /= counts[c]

    return clusters

def validate():
    correct = 0
    for input, c in zip(images, labels):
        # Find best match in cluster
        best_class = None
        min_norm = inf
        output = layer.forward(input)
        for k, cluster in enumerate(clusters):
            diff = np.linalg.norm(cluster - output)
            if diff < min_norm:
                min_norm = diff
                best_class = k

        # Validate if best class is correct
        if c == best_class:
            correct += 1
    return correct / float(len(images))


with tf.Session() as sess:
    # Run the 'init' op
    sess.run(init_op)
    sess.run(update_connections)

    for _ in range(epochs):
        result1, _ = sess.run([y, learn], feed_dict={x: [[0,0,0,0,1,0,0,0,1,0]]})
        result2, _ = sess.run([y, learn], feed_dict={x: [[1,0,0,0,1,0,0,0,0,0]]})
        result3, _ = sess.run([y, learn], feed_dict={x: [[1,0,0,0,0,1,0,0,0,0]]})
        print(result1, result2, result3)
