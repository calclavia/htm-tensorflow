import unittest
import tensorflow as tf
import numpy.testing as test

from layers import SpatialPooler

class SPTest(unittest.TestCase):
    def test_call(self):
        """
        Test forward computation
        """
        layer = SpatialPooler(4, pool_density=1)
        x = tf.placeholder(tf.float32, [1, 4], name='Input')
        y = layer(x)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            # Override permenance with a custom value
            sess.run(tf.assign(layer.p, [
                [1, 0, 1, 1],
                [0, 1, 0, 0],
                [1, 1, 1, 0],
                [0, 0, 1, 1]
            ]))

            # Compute
            result = sess.run(y, { x: [[1, 1, 0, 1]] })
            test.assert_array_equal(result, [[0, 0, 1, 0]])

    def test_train(self):
        """
        Test forward computation
        """
        layer = SpatialPooler(4, lr=0.1, pool_density=1)
        x = tf.placeholder(tf.float32, [1, 4], name='Input')
        layer.build([1, 4])
        train = layer.train(x, tf.constant([[0., 1, 0, 1]]))

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            # Override permenance with a custom value
            sess.run(tf.assign(layer.p, [
                [0.6, 0, 0.6, 0.6],
                [0, 0.6, 0, 0],
                [0.6, 0.6, 0.6, 0],
                [0, 0, 0.6, 0.6]
            ]))

            # Compute
            result = sess.run(train, { x: [[1, 1, 0, 1]] })
            # Check the new permenance
            test.assert_allclose(result, [
                [0.6, 0,   0.6, 0.7],
                [0,   0.7, 0,   0  ],
                [0.6, 0.5, 0.6, 0  ],
                [0,   0,   0.6, 0.7]
            ])

if __name__ == '__main__':
    unittest.main()
