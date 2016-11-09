import tensorflow as tf
import numpy as np
from util import *
from math import *
from layer import *

epochs = 10
dim = [784, 10]
sparsity = 0.2
learning_rate = 0.1

layer = Layer(dim, sparsity, learning_rate)

init_op = tf.initialize_all_variables()

def cluster(output_size, classes):
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

# Load MNSIT
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("data/", one_hot=False)

# Process data
input_set = [[x] for x in mnist.train.images]

with tf.Session() as sess:
    # Run the 'init' op
    sess.run(init_op)

    for epoch in range(epochs):
        print('===     Epoch ' + str(epoch) + '     ===')
        for i, x in enumerate(input_set):
            sess.run([layer.y, layer.learn], feed_dict={layer.x: x})
            progress_bar(i / float(len(input_set)))
        print()
        #print(' == Clustering ==')
