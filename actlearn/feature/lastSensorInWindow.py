from .AlFeatureTemplate import AlFeatureTemplate


class AlFeatureEventSensor(AlFeatureTemplate):

    def __init__(self, per_sensor=False):
        """
        Initialization of Template Class
        :return:
        """
        AlFeatureTemplate.__init__(self,
                                   name='lastSensorInWindow',
                                   description='Sensor ID in the current window',
                                   per_sensor=per_sensor,
                                   enabled=True,
                                   routine=None)

    def get_feature_value(self, data_list, cur_index, window_size, sensor_name=None):
        """
        If it is configured as per-sensor feature, it returns 1 if the sensor specified
        triggers the last event in the window. Otherwise returns 0.
        If it is configured as a non-per-sensor feature, it returns the index of the
        index corresponding to the dominant sensor name that triggered the last event.
        :param data_list: list of sensor data
        :param cur_index: current data record index
        :param window_size: window size
        :param sensor_name: name of sensor
        :return: a double value
        """
        sensor_label1 = data_list[cur_index]['sensor1']
        if 'sensor2' in data_list[cur_index].keys() and data_list[cur_index]['sensor2'] is not None:
            sensor_label2 = data_list[cur_index]['sensor2']
        else:
            sensor_label2 = None
        if self.per_sensor:
            if sensor_name is not None:
                if sensor_name == sensor_label1 or sensor_name == sensor_label2:
                    return 1
                else:
                    return 0
        else:
            return self.sensor_info[sensor_label1]['index']
