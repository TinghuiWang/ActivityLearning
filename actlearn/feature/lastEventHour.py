from .AlFeatureTemplate import AlFeatureTemplate
import numpy as np


class AlFeatureEventHour(AlFeatureTemplate):

    def __init__(self, normalize=False):
        """
        Initialization of Template Class
        :return:
        """
        AlFeatureTemplate.__init__(self,
                                   name='lastEventHour',
                                   description='Time of the last sensor event in window (hour)',
                                   per_sensor=False,
                                   enabled=True,
                                   routine=None)
        # Whether normalize the hour between 0 to 1
        self.normalize = normalize

    def get_feature_value(self, data_list, cur_index, window_size, sensor_name=None):
        """
        Get the hour when the last sensor event in the window occurred
        :param data_list: list of sensor data
        :param cur_index: current data record index
        :param window_size: window size
        :param sensor_name: name of sensor
        :return: a double value
        """
        if self.normalize:
            return np.float(data_list[cur_index]['datetime'].hour)/24
        else:
            return np.float(data_list[cur_index]['datetime'].hour)
