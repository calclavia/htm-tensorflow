"""
MNIST Example.
Make sure the MNIST dataset is in the data/ folder
"""
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from keras.layers import Input, Dense
from keras.models import Model
from tqdm import tqdm
import random

from math import *
from layers import SpatialPooler
from util import one_hot

epochs = 100

num_classes = 10
num_pixels = 784
pixel_bits = 4
validation_split = 0.8
input_units = num_pixels * pixel_bits
htm_units = 1024

class HTMModel:
    def __init__(self):
        pooler = SpatialPooler(htm_units, lr=1e-2)
        # Model input
        self.x = tf.placeholder(tf.float32, [1, input_units])
        self.y = pooler(self.x)
        self.train_ops = pooler.train_ops

        # Build classifier
        classifier_in = Input((htm_units,))
        classifier_out = Dense(num_classes, activation='softmax')(classifier_in)
        self.classifier = Model(classifier_in, classifier_out)
        self.classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

def main():
    # Build a model
    model = HTMModel()

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

    all_labels = [one_hot(x, num_classes) for x in mnist.train.labels[:10000]]

    num_data = int(len(all_data) * validation_split)
    num_validate = len(all_data) - num_data

    input_set = all_data[:num_data]
    input_labels = all_labels[:num_data]
    val_set = all_data[num_data:num_data+num_validate]
    val_labels = all_labels[num_data:num_data+num_validate]

    def validate(sess):
        print('Validating...')

        # Feed into HTM layer
        all_outputs = []
        for i, x in enumerate(tqdm(val_set)):
            output = sess.run(model.y, feed_dict={ model.x: x })
            all_outputs.append(output[0])

        # Feed into classifier layer
        loss, accuracy = model.classifier.evaluate(np.array(all_outputs), np.array(val_labels))
        print('Accuracy: {}'.format(accuracy))

    def train(sess):
        print('Training...')

        # Train HTM layer
        all_outputs = []
        all_labels = []

        # Shuffle input
        order = np.random.permutation(len(input_set))

        for i in tqdm(order):
            x = input_set[i]
            output, *_ = sess.run([model.y, model.train_ops], feed_dict={ model.x: x })
            all_outputs.append(output[0])
            all_labels.append(input_labels[i])

        # Train classifier
        model.classifier.fit(np.array(all_outputs), np.array(all_labels), epochs=10)

    with tf.Session() as sess:
        # Run the 'init' op
        sess.run(tf.global_variables_initializer())

        for epoch in range(epochs):
            print('=== Epoch ' + str(epoch) + ' ===')
            train(sess)
            validate(sess)

if __name__ == '__main__':
    main()
