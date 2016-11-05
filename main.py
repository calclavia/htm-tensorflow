from layer import *
from util import *
from math import *
import numpy as np

def main():
    print('Loading...')
    train_images, train_labels, images, labels = load_mnist_all(2000, 500)

    train_size = len(train_images)
    input_size = len(train_images[0])
    output_size = 10

    print('Processing...')
    train_images = [np.mat(x) for x in train_images]
    images = [np.mat(x) for x in images]

    print('Training...')
    layer = Layer(input_size, output_size)

    classes = 10
    epoch = 1

    for i in range(epoch):
        print('=== Epoch ' + str(i) + ' ===')
        for i, input in enumerate(train_images):
            output = layer.forward(input)
            layer.learn(input, output)
            progress_bar(i / float(train_size))

        print()
        print('Clustering...')
        counts = [0 for _ in range(classes)]
        clusters = [np.mat(np.zeros((1, output_size))) for _ in range(classes)]
        for input, c in zip(images, labels):
            clusters[c] += layer.forward(input)
            counts[c] += 1

        for c in range(len(clusters)):
            clusters[c] /= counts[c]

        print(layer.permanence)
        print('Validating')
        correct = 0
        for input, c in zip(images, labels):
            # Find best match in cluster
            best_class = None
            min_norm = inf
            output = layer.forward(input)
            for k, cluster in enumerate(clusters):
                diff = np.linalg.norm(cluster - output)
                if diff < min_norm:
                    min_norm = diff
                    best_class = k

            # Validate if best class is correct
            if c == best_class:
                correct += 1

        print('Accuracy:', correct / float(len(images)))

if __name__ == '__main__':
    main()
