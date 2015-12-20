from .AlFeatureTemplate import AlFeatureTemplate
from .dominantSensorFeatureRoutine import AlFeatureUpdateRoutineDominantSensor


class AlFeatureLastDominantSensor(AlFeatureTemplate):

    def __init__(self, per_sensor=False):
        """
        Initialization of Template Class
        :param per_sensor: Is this a per sensor feature?
        :return:
        """
        AlFeatureTemplate.__init__(self,
                                   name='lastDominantSensor',
                                   description='Dominant Sensor in the previous window',
                                   per_sensor=per_sensor,
                                   enabled=True,
                                   routine=AlFeatureUpdateRoutineDominantSensor())

    def get_feature_value(self, data_list, cur_index, window_size, sensor_name=None):
        """
        Get dominant sensor has two version: perSensor and regular version.
        if per_sensor is True, returns 1 with corresponding sensor Id.
        otherwise, return the index of last sensor in the window
        :param data_list: list of sensor data
        :param cur_index: current data record index
        :param window_size: window size
        :param sensor_name: name of sensor
        :return: a double value
        """
        if self.sensor_info is not None:
            dominant_sensor_label = self.routine.dominant_sensor_list[cur_index]
            if self.per_sensor:
                if sensor_name is not None:
                    if sensor_name == dominant_sensor_label:
                        return 1
                    else:
                        return 0
            else:
                return self.sensor_info[dominant_sensor_label]['index']
