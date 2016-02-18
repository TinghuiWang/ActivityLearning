from actlearn.data.mnist import *
from actlearn.utils.tile_image import tile_image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from scipy.misc import imsave
from actlearn.utils.tsne import tsne

if __name__ == '__main__':
    train_set, valid_set, test_set = mnist_load_data('../datasets/MNIST/mnist.pkl')
    (train_set_x, train_set_y) = train_set
    (valid_set_x, valid_set_y) = valid_set
    (test_set_x, test_set_y) = test_set

    train_set_rotated, valid_set_rotated, test_set_rotated = mnist_load_data('../datasets/MNIST/mnist_rotated.pkl')
    (train_set_x_rotated, train_set_y_rotated) = train_set_rotated
    (valid_set_x_rotated, valid_set_y_rotated) = valid_set_rotated
    (test_set_x_rotated, test_set_y_rotated) = test_set_rotated

    train_set_noise, valid_set_noise, test_set_noise = mnist_load_data('../datasets/MNIST/mnist_noise.pkl')
    (train_set_x_noise, train_set_y_noise) = train_set_noise
    (valid_set_x_noise, valid_set_y_noise) = valid_set_noise
    (test_set_x_noise, test_set_y_noise) = test_set_noise

    train_x_mapped = tsne(train_set_x[0:5000], 2, 50, 20.0)
    plt.figure('original')
    plt.scatter(train_x_mapped[:,0], train_x_mapped[:,1], 20, train_set_y[0:5000])
    plt.show()

    train_x_mapped_rotated = tsne(train_set_x_rotated[0:5000], 2, 50, 20.0)
    plt.figure('rotated')
    plt.scatter(train_x_mapped_rotated[:,0], train_x_mapped_rotated[:,1], 20, train_set_y[0:5000])
    plt.show()

    train_x_mapped_noise = tsne(train_set_x_noise[0:5000], 2, 50, 20.0)
    plt.figure('noise')
    plt.scatter(train_x_mapped_noise[:,0], train_x_mapped_noise[:,1], 20, train_set_y[0:5000])
    plt.show()
