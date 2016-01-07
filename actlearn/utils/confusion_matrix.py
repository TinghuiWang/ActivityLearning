import numpy as np


def get_confusion_matrix(num_classes, label, predicted):
    """
    Get Confusion Matrix
    :type num_classes: int
    :param num_class: Number of classes
    :type label: list
    :param label: Data Labels
    :param predicted: Data Labels predicted by classifier
    :return: Confusion Matrix (num_class by num_class) in numpy.array form
    """
    matrix = np.zeros((num_classes, num_classes))
    for i in range(len(label)):
        matrix[label[i]][predicted[i]] += 1
    return matrix
