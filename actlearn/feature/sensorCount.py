from .AlFeatureTemplate import AlFeatureTemplate
from .sensorCountRoutine import AlFeatureSensorCountRoutine
import numpy as np


class AlFeatureSensorCount(AlFeatureTemplate):

    def __init__(self, normalize=False):
        """
        Initialization of Template Class
        :return:
        """
        AlFeatureTemplate.__init__(self,
                                   name='sensorCount',
                                   description='Number of Events in the window related to the sensor',
                                   per_sensor=True,
                                   enabled=True,
                                   routine=AlFeatureSensorCountRoutine())
        # Normalize the number between 0 to 1
        self.normalize = normalize

    def get_feature_value(self, data_list, cur_index, window_size, sensor_name=None):
        """
        Counts the number of occurrence of the sensor specified in current window.
        :param data_list: list of sensor data
        :param cur_index: current data record index
        :param window_size: window size
        :param sensor_name: name of sensor
        :return: a double value
        """
        if self.normalize:
            return np.float(self.routine.sensor_count[sensor_name])/(window_size * 2)
        else:
            return np.float(self.routine.sensor_count[sensor_name])
