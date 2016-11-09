import tensorflow as tf

inhibit_module = tf.load_op_library('./inhibit.so')

class Layer:
    def __init__(self, dim, sparsity=0.2, learning_rate=0.1):
        self.dim = dim
        self.sparsity = sparsity
        self.learning_rate = learning_rate

        # Some input vector
        self.x = tf.placeholder(tf.float32, [1, dim[0]], name='Input')

        # Permanence of connections between neurons
        p = tf.Variable(tf.random_uniform(dim, -1, 1), name='Permanence')

        # Defines the condition to convert permanence into connections
        p_cond = tf.greater(p, 0)

        # TODO: Synapse matrix for what can possibly connect
        # Connection matrix, dependent on the permenance values
        # Computation to update the connections based on permenance
        c = tf.select(p_cond, tf.ones(dim), tf.zeros(dim))

        # Compute the activation before inhibition applied
        activation = tf.matmul(self.x, c)

        # Output vector
        self.y = inhibit_module.inhibit(activation, name='Output')

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
        self.learn = tf.assign(p, tf.add(p, delta))
