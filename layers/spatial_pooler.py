import tensorflow as tf
import numpy as np

from .layer import Layer

class SpatialPoolingLayer(Layer):
    """
    Represents the spatial pooling computation layer
    """
    def __init__(self, output_dim, sparsity=0.02, learning_rate=0.1, **kwargs):
        self.output_dim = output_dim
        self.sparsity = sparsity
        self.learning_rate = learning_rate
        self.top_k = int(self.sparsity * np.prod(self.output_dim))
        print('Spatial pooling layer with top-k=', self.top_k)
        super().__init__(**kwargs)

    def build(self, input_shape):
        # TODO: Implement potential pool matrix
        # Permanence of connections between neurons
        self.p = tf.Variable(tf.random_uniform((input_shape[1], self.output_dim), 0, 1), name='Permanence')
        super().build(input_shape)

    def call(self, x):
        # TODO: Implement potential pool matrix
        # Connection matrix, dependent on the permenance values
        # If permenance > 0.5, we are connected.
        connection = tf.to_int32(tf.round(self.p))

        # TODO: Only global inhibition is implemented.
        # TODO: Implement boosting
        # Compute the overlap score between input
        overlap = tf.matmul(tf.to_int32(x), connection)

        # Compute active mini-columns.
        # The top k activations of given sparsity activates
        # TODO: Implement stimulus threshold
        # TODO: Hard coded such that batch is not supported.
        _, act_indicies = tf.nn.top_k(overlap, k=self.top_k, sorted=False)
        act_indicies = tf.to_int64(tf.pad(tf.reshape(act_indicies, [self.top_k, 1]), [[0, 0], [1, 0]]))
        act_vals = tf.ones((self.top_k,))
        activation = tf.SparseTensor(act_indicies, act_vals, [1, self.output_dim])
        return activation

    def train(self, x, y):
        """
        Weight update
        """
        # Compute the delta permanence matrix
        # Start with the connection matrix
        # Every connection that was aligned with an input = 1. 0 otherwise.
        # Stack the input vector into n columns, where n = # of output units
        input_columned = tf.matmul(tf.transpose(x), tf.ones([1, dim[1]]))
        # Elementwise product between connection matrix and the input rows
        # Rescale values to be between -1 and 1
        alignment = (c * input_columned - 0.5) * 2
        # Only update weights to active outputs, so zero-out non-active output rows
        # We do this by constructing a diagonal matrix that scales rows to zero
        zero_out = tf.diag(y[0])
        delta = learning_rate * tf.matmul(alignment, zero_out)
        self.learn = tf.assign(self.p, tf.maximum(tf.add(self.p, delta), tf.ones_like(self.p) * -1))
