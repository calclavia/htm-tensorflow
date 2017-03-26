import tensorflow as tf

class Layer:
    """
    Represents a layer of computation
    """
    def build(self, input_shape):
        """
        Builds the computation graph
        """
        pass

    def call(self, x):
        pass

    def __call__(self, x):
        return self.call(x)
