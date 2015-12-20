try:
    import cPickle as pickle
except:
    import pickle
import os


def mnist_load_data(filename):
    """
    Load MNIST Data From Pickle file
    :param filename:
    :return:
    """
    if os.path.isfile(filename):
        mnist_file = open(filename, 'rb')
        train_set, valid_set, test_set = pickle.load(mnist_file)
        return train_set, valid_set, test_set
    else:
        print('Failed to find data file %s' % (filename))
        return None
