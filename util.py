import numpy as np

def one_hot(i, nb_classes):
    arr = np.zeros(nb_classes)
    arr[i] = 1
    return arr
