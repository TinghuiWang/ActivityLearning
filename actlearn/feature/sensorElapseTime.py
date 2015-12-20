from .AlFeatureTemplate import AlFeatureTemplate
from .sensorElapseTimeRoutine import AlFeatureUpdateSensorElapseTime
import numpy as np


class AlFeatureSensorElapseTime(AlFeatureTemplate):

    def __init__(self, normalize=False):
        """
        Initialization of Template Class
        :return:
        """
        AlFeatureTemplate.__init__(self,
                                   name='sensorElapseTime',
                                   description='Time since each sensor fired (in seconds)',
                                   per_sensor=True,
                                   enabled=True,
                                   routine=AlFeatureUpdateSensorElapseTime())
        # Normalize the number between 0 to 1
        self.normalize = normalize

    def get_feature_value(self, data_list, cur_index, window_size, sensor_name=None):
        """
        Get the time when sensor specified last fired till the end of current window
        in seconds
        :param data_list: list of sensor data
        :param cur_index: current data record index
        :param window_size: window size
        :param sensor_name: name of sensor
        :return: a double value
        """
        timedelta = data_list[cur_index]['datetime'] - self.routine.sensor_fire_log[sensor_name]
        sensor_duration = timedelta.total_seconds()
        if self.normalize:
            return np.float(sensor_duration)/(24*3600)
        else:
            return np.float(sensor_duration)
