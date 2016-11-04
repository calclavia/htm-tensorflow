from layer import *
import numpy as np

def main():
    print('Testing new layer')
    layer = Layer(10, 10)
    res = layer.forward(np.mat([1,0,0,0,1,1,1,0,0,0]))
    print(layer.connections)
    print(res)

if __name__ == '__main__':
    main()
