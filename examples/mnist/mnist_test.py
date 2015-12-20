from actlearn.data.mnist import *
import matplotlib.pyplot as plt
import matplotlib.cm as cm


if __name__ == '__main__':
    train_set, valid_set, test_set = mnist_load_data('../datasets/MNIST/mnist.pkl')
    (train_set_x, train_set_y) = train_set
    (valid_set_x, valid_set_y) = valid_set
    (test_set_x, test_set_y) = test_set

    for i in range(20):
        print train_set_y[i]
        img_matrix = train_set_x[i].reshape(28, 28)
        plt.subplot(20, 1, i)
        plt.imshow(img_matrix, cmap=cm.gray)
    plt.show()
