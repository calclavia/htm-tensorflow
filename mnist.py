"""
MNIST Example.
Make sure the MNIST dataset is in the data/ folder
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from tqdm import tqdm

from math import *
from layers import SpatialPooler

import matplotlib.pyplot as plt

epochs = 10

class Model:
    def __init__(self):
        pooler = SpatialPooler(2 ** 11)
        # Model input
        self.x = tf.placeholder(tf.bool, [1, 784], name='Input')
        self.y = pooler(self.x)
        self.train_ops = pooler.train_ops

def main():
    # Build a model
    model = Model()

    # Load MNSIT
    mnist = input_data.read_data_sets("data/", one_hot=False)

    # Process data using simple black and white encoder
    input_set = [[np.round(x)] for x in mnist.train.images]

    with tf.Session() as sess:
        # Run the 'init' op
        sess.run(tf.global_variables_initializer())

        for epoch in range(epochs):
            print('===     Epoch ' + str(epoch) + '     ===')
            for i, x in enumerate(tqdm(input_set)):
                sess.run(model.train_ops, feed_dict={ model.x: x })

            """
            print(' == Clustering ==')
            Take all the inputs and determine its cluster based on
            average values.
            ""
            counts = [0 for _ in range(dim[1])]
            clusters = [0 for _ in range(dim[1])]

            for input, label in zip(input_set, mnist.train.labels):
                clusters[label] += sess.run(layer.y, feed_dict={layer.x: input})
                counts[label] += 1

            for c in range(len(clusters)):
                clusters[c] /= counts[c]

            print(sess.run(layer.p))
            print(clusters[0])
            print(clusters[1])

            print(' == Validating == ')

            correct = 0
            for input, c in zip(mnist.validation.images, mnist.validation.labels):
                # Find best match in cluster
                best_class = None
                min_norm = inf
                output = sess.run(layer.y, feed_dict={layer.x: [input]})
                for k, cluster in enumerate(clusters):
                    diff = np.linalg.norm(cluster - output)
                    if diff < min_norm:
                        min_norm = diff
                        best_class = k

                # Validate if best class is correct
                if c == best_class:
                    correct += 1

            print('Accuracy: ', correct / float(len(mnist.validation.images)))
            """

if __name__ == '__main__':
    main()
