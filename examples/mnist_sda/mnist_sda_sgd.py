from actlearn.data.mnist import *
from actlearn.models.StackedDenoisingAutoencoder import StackedDenoisingAutoencoder
from actlearn.utils.tile_image import tile_image
from actlearn.utils.confusion_matrix import get_confusion_matrix
from actlearn.utils.classifier_performance import get_performance_array, performance_index
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import theano
import theano.tensor as T
import time
import sys


if __name__ == '__main__':
    model_file_name = 'mnist_sda.pkl'

    train_set, valid_set, test_set = mnist_load_data('../datasets/MNIST/mnist.pkl')
    (train_set_x, train_set_y) = train_set
    (valid_set_x, valid_set_y) = valid_set
    (test_set_x, test_set_y) = test_set

    train_set_x_tensor = theano.shared(np.asarray(train_set_x, dtype=theano.config.floatX), borrow=True)
    test_set_x_tensor = theano.shared(np.asarray(test_set_x, dtype=theano.config.floatX), borrow=True)
    train_set_y_tensor = T.cast(theano.shared(train_set_y, borrow=True), 'int32')
    test_set_y_tensor = T.cast(theano.shared(test_set_y, borrow=True), 'int32')

    x = T.matrix('x')
    numpy_rng = np.random.RandomState(int(time.clock()))
    model = StackedDenoisingAutoencoder(numpy_rng=numpy_rng, input=x,
                                        n_ins=train_set_x.shape[1], n_outs=10,
                                        hidden_layers_sizes=[500, 500, 500],
                                        corruption_levels=[0.1, 0.2, 0.3])
    if os.path.isfile(model_file_name):
        model.load(model_file_name)
    else:
        model.do_pretraining(data=train_set_x_tensor,
                             num_data=train_set_x.shape[0], batch_size=10,
                             learning_rate_array=[0.005, 0.005, 0.005, 0.005, 0.005, 0.005],
                             num_epochs=15)
        model.do_fine_tuning(data=train_set_x_tensor, label=train_set_y_tensor,
                             num_data=train_set_x.shape[0], batch_size=10,
                             learning_rate_array=[0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005],
                             num_epochs=36)
        # model.do_log_layer_training_only(data=train_set_x_tensor, label=train_set_y_tensor,
        #                                  num_data=train_set_x.shape[0], batch_size=10,
        #                                  learning_rate_array=[0.005, 0.005])
        model.save(model_file_name)

    num_test = test_set_x.shape[0]
    result = model.do_reconstruction(test_set_x_tensor)
    reconstruction = result[0]
    result = model.classify(test_set_x_tensor)
    class_result = result[0]
    confusion_matrix = get_confusion_matrix(10, test_set_y, class_result)
    performance = get_performance_array(10, confusion_matrix)
    sys.stdout.write('%22s\t' % ' ')
    for performance_label in performance_index:
        sys.stdout.write('%20s\t' % performance_label)
    sys.stdout.write('\n')
    num_performance = len(performance_index)
    for i in range(10):
        activity_label = ('%d' % i)
        sys.stdout.write('%22s\t' % activity_label)
        for j in range(num_performance):
            sys.stdout.write('%20.5f\t' % (performance[i][j] * 100))
        sys.stdout.write('\n')

    check_array = np.zeros(num_test)
    check_array[class_result == test_set_y] = 1
    correctness = check_array.sum() * 100 / num_test
    print(correctness)

    user_input = int(input('Input an index (max: %d; -1 for quit): ' % num_test))
    while user_input != -1:
        data_found = True
        if user_input < num_test:
            img = test_set_x[user_input].reshape(28, 28)
            img_recon = reconstruction[user_input].reshape(28, 28)
            correct_label = test_set_y[user_input]
            classify_label = class_result[user_input]
        else:
            img = None
            correct_label = ""
            classify_label = ""
            data_found = False
            print('The index should be smaller than %d. Please enter a smaller one.' % num_test)
        if data_found:
            print('class: %d, classified: %d' % (correct_label, classify_label))
            plt.figure(1)
            plt.imshow(tile_image(np.array([[img, img_recon]]),  (28, 28), (3, 3)), cmap=cm.gray)
            plt.show()
        user_input = int(input('Input an index (max: %d; -1 for quit): ' % num_test))
