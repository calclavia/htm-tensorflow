import tensorflow as tf

inhibit_module = tf.load_op_library('./inhibit.so')

class SpatialPoolingLayer(Layer):
    """
    Represents the spatial pooling computation layer
    """
    def __init__(self, output_dim, sparsity=0.02, learning_rate=0.1, **kwargs):
        self.output_dim = output_dim
        self.sparsity = sparsity
        self.learning_rate = learning_rate
        super().__init__(**kwargs)

    def build(self, input_shape):
        # TODO: Implement potential pool matrix
        # Permanence of connections between neurons
        self.p = tf.Variable(tf.random_uniform(dim, 0, 1), name='Permanence')

    def call(self, x):
        # Defines the threshold for permanences to be considered connections
        p_cond = tf.greater(self.p, 0.5)

        # TODO: Implement potential pool matrix
        # Connection matrix, dependent on the permenance values
        # Computation to update the connections based on permenance
        c = tf.select(p_cond, tf.ones(dim), tf.zeros(dim))

        # TODO: Only global inhibition is implemented.
        # TODO: Implement boosting
        # Compute the overlap score between input
        overlap = tf.matmul(x, c)

        # Compute active mini-columns.
        # TODO: Implement stimulus threshold
        self.y = inhibit_module.inhibit(overlap, name='Output')

        """
        Weight update
        """
        # Compute the delta permanence matrix
        # Start with the connection matrix
        # Every connection that was aligned with an input = 1. 0 otherwise.
        # Stack the input vector into n columns, where n = # of output units
        input_columned = tf.matmul(tf.transpose(self.x), tf.ones([1, dim[1]]))
        # Elementwise product between connection matrix and the input rows
        # Rescale values to be between -1 and 1
        alignment = (c * input_columned - 0.5) * 2
        # Only update weights to active outputs, so zero-out non-active output rows
        # We do this by constructing a diagonal matrix that scales rows to zero
        zero_out = tf.diag(self.y[0])
        delta = learning_rate * tf.matmul(alignment, zero_out)
        self.learn = tf.assign(self.p, tf.maximum(tf.add(self.p, delta), tf.ones_like(self.p) * -1))
