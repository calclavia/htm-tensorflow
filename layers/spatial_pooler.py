import tensorflow as tf
import numpy as np

from .layer import Layer

class SpatialPooler(Layer):
    """
    Represents the spatial pooling computation layer
    """
    def __init__(self, output_dim, sparsity=0.02, lr=1e-3, pool_density=0.5, **kwargs):
        """
        Args:
            - output_dim: Size of the output dimension
            - sparsity: The target sparsity to achieve
            - lr: The learning rate in which permenance is updated
            - pool_density: Percent of input a cell is connected to on average.
        """
        self.output_dim = output_dim
        self.sparsity = sparsity
        self.lr = lr
        self.pool_density = pool_density
        self.top_k = int(np.ceil(self.sparsity * np.prod(self.output_dim)))
        super().__init__(**kwargs)

    def build(self, input_shape):
        # Permanence of connections between neurons
        self.p = tf.Variable(tf.random_uniform((input_shape[1], self.output_dim), 0, 1), name='Permanence')

        # Potential pool matrix
        # Masks out the connections randomly
        rand_mask = np.random.binomial(1, self.pool_density, input_shape[1] * self.output_dim)
        pool_mask = tf.constant(np.reshape(rand_mask, [input_shape[1], self.output_dim]), dtype=tf.float32)

        # Connection matrix, dependent on the permenance values
        # If permenance > 0.5, we are connected.
        self.connection = tf.round(self.p) * pool_mask

        super().build(input_shape)

    def call(self, x):
        # TODO: Only global inhibition is implemented.
        # TODO: Implement boosting
        # Compute the overlap score between input
        overlap = tf.matmul(x, self.connection)

        # Compute active mini-columns.
        # The top k activations of given sparsity activates
        # TODO: Implement stimulus threshold
        # TODO: Hard coded such that batch is not supported.
        _, act_indicies = tf.nn.top_k(overlap, k=self.top_k, sorted=False)
        act_indicies = tf.to_int64(tf.pad(tf.reshape(act_indicies, [self.top_k, 1]), [[0, 0], [1, 0]]))
        act_vals = tf.ones((self.top_k,))
        activation = tf.SparseTensor(act_indicies, act_vals, [1, self.output_dim])
        print(activation,  self.output_dim)
        # TODO: Keeping it as a sparse tensor is more efficient.
        activation = tf.sparse_tensor_to_dense(activation, validate_indices=False)
        return activation

    def train(self, x, y):
        """
        Weight update using Hebbian learning rule.

        For each active SP mini-column, we reinforce active input connections
        by increasing the permanence, and punish inactive connections by
        decreasing the permanence.
        We only want to modify permances of connections in active mini-columns.
        Ignoring all non-connections.
        Connections are clipped between 0 and 1.
        """
        # Construct a binary connection matrix with all non-active mini-columns
        # masked to zero. This contains all connections to active units.
        # Multiply using broadcasting behavior to mask out inactive units.
        active_cons = y * self.connection

        # Shift input X from 0, 1 to -1, 1.
        x_shifted = 2 * x - 1
        # Compute delta matrix, which contains -1 for all connections to punish
        # and 1 for all connections to reinforce. Use broadcasting behavior.
        delta = tf.transpose(x_shifted * tf.transpose(active_cons))

        # Apply learning rate multiplier
        new_p = tf.clip_by_value(self.p + self.lr * delta, 0, 1)

        # Create train op
        train_op = tf.assign(self.p, new_p)
        return train_op
