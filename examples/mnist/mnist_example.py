from actlearn.data.mnist import *
from actlearn.utils.tile_image import tile_image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from scipy.misc import imsave


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

    img_array = np.zeros((5, 6, 28, 28))

    for label in range(10):
        i = 0
        while train_set_y[i] != label:
            i += 1
        start_col = (label % 2) * 3
        end_col = start_col + 3
        img_array[label/2, start_col:end_col, :, :] = [
            train_set_x[i].reshape(28, 28),
            train_set_x_rotated[i].reshape(28, 28),
            train_set_x_noise[i].reshape(28, 28)
        ]

    img_rendered = tile_image(img_array, (28, 28), (3, 3))
    imsave('MNIST_example.png', img_rendered)

    plt.figure(1)
    plt.imshow(img_rendered, cmap=cm.gray)
    plt.show()
