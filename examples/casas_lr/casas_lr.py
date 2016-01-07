import os
import sys
import logging.config
import theano
import theano.tensor as T
import cPickle as pickle
import numpy as np
from actlearn.data.casas import load_casas_from_file
from actlearn.training_algorithms.cross_validation import run_cross_validation
from actlearn.models.LogisticRegression import LogisticRegression
from actlearn.utils.confusion_matrix import get_confusion_matrix
from actlearn.utils.classifier_performance import performance_index
from actlearn.data.AlFeature import AlFeature


def run_test(act_feature):
    """
    :type act_feature: AlFeature
    :param act_feature:
    :return:
    """
    x = act_feature.x
    y = act_feature.y
    num_classes = act_feature.num_enabled_activities
    input_x = T.matrix('x')
    # Fold number is set to 3
    num_fold = 3
    model = LogisticRegression(input=input_x, n_in=x.shape[1], n_out=num_classes)

    # After Get the feature, run 3-fold cross validation and search for best hyper-parameters
    performance = run_cross_validation(n=num_fold, num_classes=num_classes,
                                       data=x, label=y,
                                       train_func=casas_train, test_func=casas_test,
                                       model=model)
    sys.stdout.write('%22s\t' % ' ')
    for performance_label in performance_index:
        sys.stdout.write('%20s\t' % performance_label)
    sys.stdout.write('\n')
    num_performance = len(performance_index)
    for i in range(num_classes):
        activity_label = act_feature.get_activity_by_index(i)
        sys.stdout.write('%22s\t' % activity_label)
        for j in range(num_performance):
            sys.stdout.write('%20.5f\t' % (performance[i][j] * 100))
        sys.stdout.write('\n')


def casas_train(x, y, model):
    """
    Training Function using Logistic Regression
    :param x: numpy.array training data
    :param y: numpy.array training labels
    :param model: an model to be trained (in this case: Logistic Regression Object)
    :return: None
    """
    assert(type(model) == LogisticRegression)
    assert(x.shape[0] == y.shape[0])

    x_tensor = theano.shared(np.asarray(x, dtype=theano.config.floatX), borrow=True)
    y_tensor = T.cast(theano.shared(y, borrow=True), 'int32')

    model.do_training_sgd(data=x_tensor, label=y_tensor,
                          num_data=x.shape[0], batch_size=10,
                          learning_rate_array=[0.1, 0.1], num_epochs=10)


def casas_test(x, y, num_classes, model):
    """
    Test Trained Logistic Regression Model
    :param x: numpy.array training data
    :type y: numpy.array
    :param y: numpy.array training labels
    :param num_classes: integer number of enabled classes
    :param model: an model to be trained (in this case: Logistic Regression Object)
    :return: numpy.array (confusion matrix)
    """
    assert(type(model) == LogisticRegression)
    x_tensor = theano.shared(np.asarray(x, dtype=theano.config.floatX), borrow=True)
    result = model.classify(x_tensor)
    predicted_y = result[0]
    confusion_matrix = get_confusion_matrix(num_classes=num_classes, label=y, predicted=predicted_y)
    return confusion_matrix


if __name__ == '__main__':
    # Set current directory to local directory
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    # Go through all bosch datasets
    datasets = ['b1']
    for datafile in datasets:
        feature_filename = 'feature_' + datafile + '.pkl'
        # Looking for processed feature data
        if os.path.exists(feature_filename):
            feature_file = open(feature_filename, mode='r')
            feature_dict = pickle.load(feature_file)
            feature = AlFeature()
            feature.load_from_dict(feature_dict)
        else:
            feature = load_casas_from_file(datafile, datafile + '.translate')
            feature_file = open(feature_filename, mode='w')
            pickle.dump(feature.export_to_dict(), feature_file, protocol=-1)
        feature_file.close()
        run_test(feature)
