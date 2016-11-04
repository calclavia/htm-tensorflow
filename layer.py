import numpy as np
from util import PriorityQueue
from math import *

class Layer:
    def __init__(self, in_size, out_size, threshold=0.5):
        self.in_size = in_size
        self.out_size = out_size
        self.threshold = threshold
        # 2% of the output should be active
        self.max_active = ceil(out_size * 0.02)

        # Binary activation
        self.bin_threshold = np.vectorize(lambda x: 1 if x > self.threshold else 0)

        # Learned parameters
        dim = (out_size, in_size)
        self.synapse = np.mat(np.zeros(dim), dtype='bool')
        # Float between 0 and 1 specifying how strong the connections are.
        self.permanence = np.mat(np.random.uniform(0, 1, (dim)))
        self.update_connections()

    def forward(self, input):
        """
        Perform spatial pooling algorithm.

        Parameters:
            - input: Input boolean vector
        """
        # TODO: Boost the active ones?
        overlap_scores = input * self.connections

        # Apply global inhibition (pick top n scores)
        inhibited = self.global_inhibit(overlap_scores)
        return self.bin_threshold(inhibited)

    def global_inhibit(self, overlap_scores):
        # Set of outputs to keep
        keep = PriorityQueue()

        for i in range(self.out_size):
            score = overlap_scores[0, i]

            if keep.empty() or score > keep.peek():
                # We want this score to be kept

                if keep.size() < self.max_active:
                    keep.update(i, score)
                else:
                    # Replace the lowest score in keep with this one
                    keep.pushpop(i, score)

        # Set all indices not kept to 0
        keep_indicies = {i for _, _, i in keep.heap}

        for i in range(self.out_size):
            if i not in keep_indicies:
                overlap_scores[0, i] = 0

        return overlap_scores

    def update_connections(self):
        """
        Recalculate the connected synapses based on permanence and a global
        threshold
        """
        # 1 if connected. 0 if disconnected
        self.connections = self.bin_threshold(self.permanence)
