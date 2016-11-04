import struct

def chunk(l, size):
    """
    Chunks a list by size
    """
    return [l[x:x+size] for x in range(0, len(l), size)]

def load_mnist(images, labels, limit):
    """
    Takes two filenames "images" and "labels" and reads
    them, returning a tuple containing a list of images
    to their corresponding labels (indexed).
    """
    # Read images
    with open(images, 'rb') as f_images:
        # Read the image data header
        f_magic, size, rows, cols = struct.unpack(">IIII", f_images.read(16))
        # Extract raw image data as unsigned bytes
        img_raw_data = list(f_images.read())

    # Chunk each image separately, and limit the # of images
    dimension = rows * cols
    images = chunk(img_raw_data, dimension)[:limit]

    # Read labels
    with open(labels, 'rb') as f_labels:
        f_magic, size = struct.unpack(">II", f_labels.read(8))
        # Extract raw image data as unsigned bytes
        labels = list(f_labels.read())[:limit]

    return (images, labels, dimension)

def load_mnist_all(train_n=20000, validate_n=2000):
    train_images, train_labels, *_ = load_mnist('data/train-images-idx3-ubyte', 'data/train-labels-idx1-ubyte', train_n)
    images, labels, *_ = load_mnist('data/t10k-images-idx3-ubyte', 'data/t10k-labels-idx1-ubyte', validate_n)
    return (train_images, train_labels, images, labels)


def progress_bar(percent, size=20):
    """
    Print progress bar
    """
    sys.stdout.write('\r')
    sys.stdout.write(("[%-" + str(size) + "s] %d%%") % ('=' * int(size * percent), 100 * percent))
    sys.stdout.flush()
