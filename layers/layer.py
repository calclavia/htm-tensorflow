import tensorflow as tf

class Layer:
    """
    Represents a layer of computation
    """
    def __init__(self):
        self.is_built = False

    def build(self, input_shape):
        """
        Builds variables
        """
        assert self.is_built == False
        self.is_built = True

    def call(self, x):
        pass

    def __call__(self, x):
        if not self.is_built:
            self.build(x.get_shape().as_list())

        return self.call(x)
