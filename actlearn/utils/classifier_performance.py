import numpy as np


performance_index = ['true_positive', 'true_negative', 'false_positive', 'false_negative',
                     'accuracy', 'misclassification', 'recall', 'false positive rate',
                     'specificity', 'precision', 'prevalence']


def get_performance_array(num_classes, confusion_matrix):
    """
    Gets performance array for each class
    0 - True_Positive: number of samples that belong to class and classified correctly
    1 - True_Negative: number of samples that correctly classified as not belonging to class
    2 - False_Positive: number of samples that belong to class and not classified correctly
    3 - False_Negative: number of samples that do not belong to class but classified as class
    4 - Accuracy: Overall, how often is the classifier correct? (TP + TN) / (TP + TN + FP + FN)
    5 - Misclassification: Overall, how often is it wrong? (FP + FN) / (TP + TN + FP + FN)
    6 - Recall: When it's actually yes, how often does it predict yes? TP / (TP + FN)
    7 - False Positive Rate: When it's actually no, how often does it predict yes? FP / (FP + TN)
    8 - Specificity: When it's actually no, how often does it predict no? TN / (FP + TN)
    9 - Precision: When it predicts yes, how often is it correct? TP / (TP + FP)
    10 - Prevalence: How often does the yes condition actually occur in our sample? Total(class) / Total(samples)
    :param num_classes: Number of classes
    :param confusion_matrix: Confusion Matrix (numpy array of num_class by num_class)
    :return:
    """
    assert(confusion_matrix.shape[0] == confusion_matrix.shape[1])
    assert(num_classes == confusion_matrix.shape[0])

    performance = np.zeros((num_classes, 11), dtype=float)

    for i in range(num_classes):
        true_positive = confusion_matrix[i][i]
        true_negative = np.sum(confusion_matrix)\
            - np.sum(confusion_matrix[i][:])\
            - np.sum(confusion_matrix[:][i])\
            + confusion_matrix[i][i]
        false_positive = np.sum(confusion_matrix[:][i]) - confusion_matrix[i][i]
        false_negative = np.sum(confusion_matrix[i][:]) - confusion_matrix[i][i]
        performance[i][0] = true_positive
        performance[i][1] = true_negative
        performance[i][2] = false_positive
        performance[i][3] = false_negative
        # Accuracy: (TP + TN) / (TP + TN + FP + FN)
        performance[i][4] = (true_positive + true_negative)\
            / (true_positive + true_negative + false_positive + false_negative) * 100
        # Mis-classification: (FP + FN) / (TP + TN + FP + FN)
        performance[i][5] = (false_positive + false_negative)\
            / (true_positive + true_negative + false_positive + false_negative) * 100
        # Recall: TP / (TP + FN)
        if true_positive + false_negative == 0:
            performance[i][6] = float('nan')
        else:
            performance[i][6] = true_positive / (true_positive + false_negative) * 100
        # False Positive Rate: FP / (FP + TN)
        if false_positive + true_negative == 0:
            performance[i][7] = float('nan')
        else:
            performance[i][7] = false_positive / (false_positive + true_negative) * 100
        # Specificity: TN / (FP + TN)
        if false_positive + true_negative == 0:
            performance[i][8] = float('nan')
        else:
            performance[i][8] = true_negative / (false_positive + true_negative) * 100
        # Precision: TP / (TP + FP)
        if true_positive + false_positive == 0:
            performance[i][9] = float('nan')
        else:
            performance[i][9] = true_positive / (true_positive + false_positive) * 100
        # prevalence
        performance[i][10] = (true_positive + false_negative)\
            / (true_positive + true_negative + false_positive + false_negative) * 100
    return performance
