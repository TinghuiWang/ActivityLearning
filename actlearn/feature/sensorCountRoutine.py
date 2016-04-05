from .AlFeatureUpdateRoutineTemplate import AlFeatureUpdateRoutineTemplate


class AlFeatureSensorCountRoutine (AlFeatureUpdateRoutineTemplate):

    def __init__(self):
        """
        Initialization of Template Class
        :return:
        """
        AlFeatureUpdateRoutineTemplate.__init__(
            self,
            name='SensorCountRoutine',
            description='Count Occurrence of all sensors in current event window',
        )
        # Dominant Sensor
        self.sensor_count = {}

    def update(self, data_list, cur_index, window_size):
        """
        Record the number of occurrence of each sensor in the sensor count
        dictionary
        :param data_list: list of sensor data
        :param cur_index: current data record index
        :param window_size: window size
        :return: None
        """
        self.sensor_count = {}
        for sensor_label in self.sensor_info.keys():
            if self.sensor_info[sensor_label]['enable']:
                self.sensor_count[sensor_label] = 0
        for index in range(0, window_size):
            if data_list[cur_index - index]['sensor1'] in self.sensor_count.keys():
                self.sensor_count[data_list[cur_index - index]['sensor1']] += 1
            if 'sensor2' in data_list[cur_index - index].keys() and data_list[cur_index - index]['sensor2'] is not None:
                if data_list[cur_index - index]['sensor2'] in self.sensor_count.keys():
                    self.sensor_count[data_list[cur_index - index]['sensor2']] += 1

    def clear(self):
        """
        {inherit}
        """
        self.sensor_count = {}
