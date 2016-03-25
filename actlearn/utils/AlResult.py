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
        """
        :param result_name:
        :param data_fname:
        :param mode: single_shot, by_week, by_day
        :return:
        """
        cur_time = time.time()
        self.data = {
            'result_name': result_name,
            'data_fname': data_fname,
            'created_time': cur_time,
            'modified_time': cur_time,
            'mode': mode,
            'records': collections.OrderedDict()
        }
        return

    def get_mode(self):
        """
        Get Result Mode
        :return:
        """
        return self.data['mode']

    def get_num_records(self):
        """
        Get the length of result records in current instance
        :return:
        """
        return len(self.data['records'])

    def get_record_keys(self):
        """
        Get List of keys to all the records
        :return:
        """
        return self.data['records'].keys()

    def get_name(self):
        """
        Get the name of current result records
        :return:
        """
        return self.data['result_name']

    def add_record(self, model, key='single_shot', confusion_matrix=None,
                   per_class_performance=None, overall_performance=None):
        """
        :param model:
        :param confusion_matrix:
        :param key:
        :param per_class_performance:
        :param overall_performance:
        :return:
        """
        cur_result = {
            'model': model,
            'confusion_matrix': confusion_matrix,
            'per_class_performance': per_class_performance,
            'overall_performance': overall_performance
        }
        self.data['records'][key] = cur_result

    def get_record_by_key(self, key):
        """
        Get result corresponding to specific key
        :param key:
        :return:
        """
        if key in self.data['records'].keys():
            return self.data['records'][key]
        else:
            return None

    def save_to_file(self, fname):
        """
        :type fname: str
        :param fname: file name
        :return:
        """
        fp = open(fname, 'w+')
        pickle.dump(self.data, fp, protocol=-1)
        fp.close()

    def load_from_file(self, fname):
        """
        :type fname: str
        :param fname: file name
        :return:
        """
        if os.path.exists(fname):
            fp = open(fname, 'r')
            self.data = pickle.load(fp)
            fp.close()
            # Due to naming change, result is renamed to records.
            # The following piece of codes update the naming automatically when detected
            if 'result' in self.data.keys():
                self.data['records'] = self.data.pop('result', None)
                self.save_to_file(fname)
