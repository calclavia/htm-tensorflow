import numpy as np

class Layer:
    def __init__(self, in_size, out_size):
        self.in_size = in_size
        self.out_size = out_size

        # Learned parameters
        dim = (out_size, in_size)
        self.synapse = np.matrix(np.zeros(dim), dtype='bool')
        # Float between 0 and 1 specifying how strong the connections are.
        self.permanence = np.matrix(np.random.uniform(0, 1, (dim)))
        # 1 if connected. 0 if disconnected
        self.connections = np.matrix(np.zeros(dim), dtype='bool')

    def forward(self, input):
        """
        Perform spatial pooling algorithm.

        Parameters:
            - input: Input boolean vectogir
        """
        pass
