from actlearn.data.mnist import *
from scipy import ndimage
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.cm as cm
try:
   import cPickle as pickle
except:
   import pickle


if __name__ == '__main__':
    train_set, valid_set, test_set = mnist_load_data('../datasets/MNIST/mnist.pkl')
    (train_set_x, train_set_y) = train_set
    (valid_set_x, valid_set_y) = valid_set
    (test_set_x, test_set_y) = test_set

    train_set_x_rotated = np.zeros(train_set_x.shape)
    valid_set_x_rotated = np.zeros(valid_set_x.shape)
    test_set_x_rotated = np.zeros(test_set_x.shape)

    # Create Rotation Set
    for i in range(train_set_x.shape[0]):
        rotate_angle = random.randint(0,45)
        img = train_set_x[i].reshape(28, 28)
        img_rotated = ndimage.rotate(img, rotate_angle, reshape=False)
        train_set_x_rotated[i] = img_rotated.reshape(28*28)

    for i in range(valid_set_x.shape[0]):
        rotate_angle = random.randint(0,45)
        img = valid_set_x[i].reshape(28, 28)
        img_rotated = ndimage.rotate(img, rotate_angle, reshape=False)
        valid_set_x_rotated[i] = img_rotated.reshape(28*28)

    for i in range(test_set_x.shape[0]):
        rotate_angle = random.randint(0,45)
        img = test_set_x[i].reshape(28, 28)
        img_rotated = ndimage.rotate(img, rotate_angle, reshape=False)
        test_set_x_rotated[i] = img_rotated.reshape(28*28)

    out_file = open('../datasets/MNIST/mnist_rotated.pkl', 'w')
    train_set_rotated = (train_set_x_rotated, train_set_y)
    valid_set_rotated = (valid_set_x_rotated, valid_set_y)
    test_set_rotated = (test_set_x_rotated, test_set_y)
    pickle.dump((train_set_rotated, valid_set_rotated, test_set_rotated), out_file, protocol=-1)
    out_file.close()

    # Create Noise Set
    train_set_x_noise = np.zeros(train_set_x.shape)
    valid_set_x_noise = np.zeros(valid_set_x.shape)
    test_set_x_noise = np.zeros(test_set_x.shape)

    for i in range(train_set_x.shape[0]):
        background = np.random.uniform(0.0, 1.0, (28, 28))
        img = train_set_x[i].reshape(28, 28)
        img += background
        img[img > 1] = 1
        train_set_x_noise[i] = img.reshape(28*28)

    for i in range(valid_set_x.shape[0]):
        background = np.random.uniform(0.0, 1.0, (28, 28))
        img = valid_set_x[i].reshape(28, 28)
        img += background
        img[img > 1] = 1
        valid_set_x_noise[i] = img.reshape(28*28)

    for i in range(test_set_x.shape[0]):
        background = np.random.uniform(0.0, 1.0, (28, 28))
        img = test_set_x[i].reshape(28, 28)
        img += background
        img[img > 1] = 1
        test_set_x_noise[i] = img.reshape(28*28)

    out_file = open('../datasets/MNIST/mnist_noise.pkl', 'w')
    train_set_noise = (train_set_x_noise, train_set_y)
    valid_set_noise = (valid_set_x_noise, valid_set_y)
    test_set_noise = (test_set_x_noise, test_set_y)
    pickle.dump((train_set_noise, valid_set_noise, test_set_noise), out_file, protocol=-1)
    out_file.close()

    img = train_set_x[560].reshape(28, 28)
    img_rotated = train_set_x_noise[560].reshape(28, 28)
    fig = plt.figure(1)
    plt.subplot(1,3,1)
    plt.imshow(img, cmap=cm.gray)
    plt.subplot(1,3,2)
    plt.imshow(img_rotated, cmap=cm.gray)
    plt.show()
