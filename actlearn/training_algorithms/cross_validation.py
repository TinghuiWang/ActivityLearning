import numpy as np
from actlearn.utils.classifier_performance import get_performance_array, performance_index


def run_cross_validation(n, num_classes, data, label, train_func, test_func, model, **kwargs):
    """
    Run n-fold cross validation template
    :param n: Number of folds
    :param num_classes: total number of classes
    :param data: Data Array (numpy.array)
    :param label: Corresponding Labels (numpy.array)
    :param train_func: Training Function Pointer
    :param test_func: Test Function Pointer
    :param args: Shared arguments for test and train function
    :param model: Learning Models
    :param kwargs: Shared key-argument pairs for test and train functions
    :return: Average error rate
    """
    num_samples = data.shape[0]
    training_stat = np.zeros((n, num_classes, len(performance_index)), dtype=np.float)
    for i in range(n):
        train_index = []
        test_index = []
        for j in range(num_samples):
            if j % n == i:
                test_index.append(j)
            else:
                train_index.append(j)

        test_set_x = data[test_index]
        test_set_y = label[test_index]
        train_set_x = data[train_index]
        train_set_y = label[train_index]

        model.clear()

        train_func(x=train_set_x, y=train_set_y, model=model, **kwargs)
        # Multi-class report of a test function should be a m by m matrix
        # where m is the number of target classes
        # Row is (IS Class i)
        # Column is (Declared as class i)
        result = test_func(x=test_set_x, y=test_set_y, num_classes=num_classes, model=model, **kwargs)

        # Calculate Correctness
        training_stat[i] = get_performance_array(num_classes=num_classes, confusion_matrix=result)

    return np.average(training_stat, axis=0)
