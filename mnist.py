"""
MNIST Example.
Make sure the MNIST dataset is in the data/ folder
"""
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tqdm import tqdm

from math import *
from layers import SpatialPooler

import matplotlib.pyplot as plt

epochs = 10

num_pixels = 784
pixel_bits = 4
validation_split = 0.8

class Model:
    def __init__(self):
        pooler = SpatialPooler(1024, lr=1e-2)
        # Model input
        self.x = tf.placeholder(tf.float32, [1, num_pixels * pixel_bits])
        self.y = pooler(self.x)
        self.train_ops = pooler.train_ops

def one_hot(i, nb_classes):
    arr = np.zeros(nb_classes)
    arr[i] = 1
    return arr

def main():
    # Build a model
    model = Model()

    # Load MNSIT
    print('Loading data...')
    mnist = input_data.read_data_sets("data/", one_hot=False)

    # Process data using simple greyscale encoder
    all_data = []

    print('Processing data...')
    for img in tqdm(mnist.train.images[:10000]):
        img_data = []
        for pixel in img:
            # one-hot representation
            index = min(int(pixel * pixel_bits), pixel_bits - 1)
            img_data += list(one_hot(index, pixel_bits))
        all_data.append([img_data])

    all_labels = mnist.train.labels[:10000]

    num_data = int(len(all_data) * validation_split)
    num_validate = len(all_data) - num_data

    input_set = all_data[:num_data]
    val_set = all_data[num_data:num_data+num_validate]
    val_labels = all_labels[num_data:num_data+num_validate]

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
            x = all_data[label_indicies[label]]
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

    with tf.Session() as sess:
        # Run the 'init' op
        sess.run(tf.global_variables_initializer())

        train(sess)

if __name__ == '__main__':
    main()
