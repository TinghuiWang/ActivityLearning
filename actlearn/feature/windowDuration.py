from .AlFeatureTemplate import AlFeatureTemplate
import numpy as np


class AlFeatureWindowDuration(AlFeatureTemplate):
    def __init__(self, normalize=False):
        """
        Initialization of Template Class
        :return:
        """
        AlFeatureTemplate.__init__(self,
                                   name='windowDuration',
                                   description='Duration of current window in seconds',
                                   per_sensor=False,
                                   enabled=True,
                                   routine=None)
        self.is_valid = True
        # Whether normalize the hour between 0 to 1
        self.normalize = normalize

    def get_feature_value(self, data_list, cur_index, window_size, sensor_name=None):
        """
        Calculate the duration of current window in seconds
        :param data_list: list of sensor data
        :param cur_index: current data record index
        :param window_size: window size
        :param sensor_name: name of sensor
        :return: a double value
        """
        self.is_valid = True
        timedelta = data_list[cur_index]['datetime'] - data_list[cur_index - window_size + 1]['datetime']
        window_duration = timedelta.total_seconds()
        if window_duration > 3600 * 12:
            self.is_valid = False
            # Window Duration is greater than a day - not possible
            # print('Warning: curIndex: %d; windowSize: %d; windowDuration: %f' %
            # (curIndex, windowSize, window_duration))
            window_duration -= 3600 * 12 * (int(window_duration) / (3600 * 12))
            # print('Fixed window duration %f' % window_duration)
            if data_list[cur_index]['datetime'].month != data_list[cur_index - 1]['datetime'].month or \
                    data_list[cur_index]['datetime'].day != data_list[cur_index - 1]['datetime'].day:
                date_advanced = (data_list[cur_index]['datetime'] - data_list[cur_index - 1]['datetime']).days
                hour_advanced = data_list[cur_index]['datetime'].hour - data_list[cur_index - 1]['datetime'].hour
                print('line %d - %d: %s' % (cur_index, cur_index + 1, data_list[cur_index - 1]['datetime'].isoformat()))
                print('Date Advanced: %d; hour gap: %d' % (date_advanced, hour_advanced))
        if self.normalize:
            return np.float(window_duration) / (3600 * 12)
        else:
            return np.float(window_duration)

    def is_value_valid(self):
        return self.is_valid
