import tensorflow as tf

class Layer:
    """
    Represents a layer of computation
    """
    def __init__(self):
        self.is_built = False
        self.train_ops = []

    def build(self, input_shape):
        """
        Builds variables
        """
        assert self.is_built == False
        self.is_built = True

    def call(self, x):
        """
        Returns the tensor output of this layer
        """
        pass

    def train(self, x, y):
        """
        Returns the train op of this layer
        """

    def __call__(self, x):
        if not self.is_built:
            self.build(x.get_shape().as_list())

        y = self.call(x)
        self.train_ops.append(self.train(x, y))
        return y
