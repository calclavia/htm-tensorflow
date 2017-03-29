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
        pooler = SpatialPooler(2 ** 11, lr=1e-2)
        # Model input
        self.x = tf.placeholder(tf.float32, [1, 784], name='Input')
        self.y = pooler(self.x)
        self.train_ops = pooler.train_ops

def main():
    # Build a model
    model = Model()

    # Load MNSIT
    mnist = input_data.read_data_sets("data/", one_hot=False)

    # Process data using simple greyscale encoder
    all_data = mnist.train.images
    all_labels = mnist.train.labels

    total_num_data = len(all_data) // 2
    num_data = int(total_num_data * 0.8)
    num_validate = total_num_data - num_data

    input_set = [[np.round(x)] for x in all_data[:num_data]]
    val_set = [[np.round(x)] for x in all_data[num_data:num_data+num_validate]]
    val_labels = mnist.train.labels[num_data:num_data+num_validate]

    # Find one of each label
    label_indicies = {}

    for i, label in enumerate(mnist.train.labels):
        if label not in label_indicies:
            label_indicies[label] = i

        if len(label_indicies) == 10:
            break

    def validate(sess):
        print('Validating...')
        # Retrieve label mapping
        label_mappings = {}

        for label in range(10):
            x = [all_data[label_indicies[label]]]
            label_mappings[label] = sess.run(model.y, feed_dict={ model.x: x })

        correct = 0
        total = 0
        for i, x in enumerate(tqdm(val_set)):
            result = sess.run(model.y, feed_dict={ model.x: x })

            # Nearest neighbor
            closest_label = None
            closest_dist = float('inf')

            for label, mapping in label_mappings.items():
                diff = mapping - result
                dist = np.dot(diff[0], diff[0])
                if dist < closest_dist:
                    closest_label = label
                    closest_dist = dist

            if closest_label == val_labels[i]:
                correct += 1
            total += 1

        print('Accuracy: {}'.format(correct / total))

    def train(sess):
        for epoch in range(epochs):
            print('=== Epoch ' + str(epoch) + ' ===')
            validate(sess)

            # Shuffle input
            order = np.random.permutation(len(input_set))

            # Train HTM layer
            for i in tqdm(order):
                x = input_set[i]
                sess.run(model.train_ops, feed_dict={ model.x: x })

    with tf.device('cpu:0'), tf.Session() as sess:
        # Run the 'init' op
        sess.run(tf.global_variables_initializer())

        train(sess)

if __name__ == '__main__':
    main()
