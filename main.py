from layer import *
from util import *
import numpy as np

def main():
    print('Loading data')
    train_images, train_labels, images, labels = load_mnist_all()

    print('Processing data')
    train_images = [np.matrix(x) for x in train_images]

    train_size = len(train_images)
    input_size = len(train_images[0])

    print('Testing new layer')
    layer = Layer(input_size, 10)

    epoch = 10

    for i in range(epoch):
        for i, input in enumerate(train_images):
            output = layer.forward(input)
            layer.learn(input, output)
            progress_bar(i / float(train_size))

if __name__ == '__main__':
    main()
