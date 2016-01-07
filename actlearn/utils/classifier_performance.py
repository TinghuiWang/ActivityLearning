import numpy as np


performance_index = ['accuracy', 'misclassification', 'recall', 'false positive rate',
                     'specificity', 'precision', 'prevalence']


def get_performance_array(num_classes, confusion_matrix):
    """
    Gets performance array for each class
    0 - Accuracy: Overall, how often is the classifier correct? (TP + TN) / (TP + TN + FP + FN)
    1 - Misclassification: Overall, how often is it wrong? (FP + FN) / (TP + TN + FP + FN)
    2 - Recall: When it's actually yes, how often does it predict yes? TP / (TP + FN)
    3 - False Positive Rate: When it's actually no, how often does it predict yes? FP / (FP + TN)
    4 - Specificity: When it's actually no, how often does it predict no? TN / (FP + TN)
    5 - Precision: When it predicts yes, how often is it correct? TP / (TP + FP)
    6 - Prevalence: How often does the yes condition actually occur in our sample? Total(class) / Total(samples)
    :param num_classes: Number of classes
    :param confusion_matrix: Confusion Matrix (numpy array of num_class by num_class)
    :return:
    """
    assert(confusion_matrix.shape[0] == confusion_matrix.shape[1])
    assert(num_classes == confusion_matrix.shape[0])

    performance = np.zeros((num_classes, 7), dtype=float)

    for i in range(num_classes):
        true_positive = confusion_matrix[i][i]
        true_negative = np.sum(confusion_matrix)\
            - np.sum(confusion_matrix[i][:])\
            - np.sum(confusion_matrix[:][i])\
            + confusion_matrix[i][i]
        false_positive = np.sum(confusion_matrix[:][i]) - confusion_matrix[i][i]
        false_negative = np.sum(confusion_matrix[i][:]) - confusion_matrix[i][i]
        # Accuracy: (TP + TN) / (TP + TN + FP + FN)
        performance[i][0] = (true_positive + true_negative)\
            / (true_positive + true_negative + false_positive + false_negative)
        # Mis-classification: (FP + FN) / (TP + TN + FP + FN)
        performance[i][1] = (false_positive + false_negative)\
            / (true_positive + true_negative + false_positive + false_negative)
        # Recall: TP / (TP + FN)
        performance[i][2] = true_positive / (true_positive + false_negative)
        # False Positive Rate: FP / (FP + TN)
        performance[i][3] = false_positive / (false_positive + true_negative)
        # Specificity: TN / (FP + TN)
        performance[i][4] = true_negative / (false_positive + true_negative)
        # Precision: TP / (TP + FP)
        performance[i][5] = true_positive / (true_positive + false_positive)
        # prevalence
        performance[i][6] = (true_positive + false_negative)\
            / (true_positive + true_negative + false_positive + false_negative)
    return performance
