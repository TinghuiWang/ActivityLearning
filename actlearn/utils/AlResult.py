import cPickle as pickle
import time
import os
import collections


class AlResult:
    """
    AlResult is a class that stores results of a learning run.
    It may be a single-shot run or a time-based analysis
    The result structure holds the parameters for the model as well as
    the evaluation result for easy plot
    """
    def __init__(self, result_name='', data_fname='', mode='single_shot'):
        cur_time = time.time()
        self.data = {
            'result_name': result_name,
            'data_fname': data_fname,
            'created_time': cur_time,
            'modified_time': cur_time,
            'mode': mode,
            'result': collections.OrderedDict()
        }
        return

    def add_result(self, model, key='single_shot', confusion_matrix=None,
                   per_class_performance=None, overall_performance=None):
        """
        :param model:
        :param confusion_matrix:
        :param key:
        :return:
        """
        cur_result = {
            'model': model,
            'confusion_matrix': confusion_matrix,
            'per_class_performance': per_class_performance,
            'overall_performance': overall_performance
        }
        self.data['result'][key] = cur_result

    def get_result_by_key(self, key):
        """
        Get result corresponding to specific key
        :param key:
        :return:
        """
        if key in self.data['result'].keys():
            return self.data['result'][key]
        else:
            return None

    def visualize(self):
        """
        python.wx GUI to visualize the data
        :return:
        """

    def save_result(self, fname):
        """
        :type fname: str
        :param fname: file name
        :return:
        """
        fp = open(fname, 'w+')
        pickle.dump(self.data, fp, protocol=-1)
        fp.close()

    def load_result(self, fname):
        """
        :type fname: str
        :param fname: file name
        :return:
        """
        if os.path.exists(fname):
            fp = open(fname, 'r')
            self.data = pickle.load(fp)
            fp.close()
