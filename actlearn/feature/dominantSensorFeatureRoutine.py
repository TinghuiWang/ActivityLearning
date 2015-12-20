from .AlFeatureUpdateRoutineTemplate import AlFeatureUpdateRoutineTemplate


class AlFeatureUpdateRoutineDominantSensor (AlFeatureUpdateRoutineTemplate):

    def __init__(self):
        """
        Initialization of Template Class
        :return:
        """
        AlFeatureUpdateRoutineTemplate.__init__(
            self,
            name='DominantSensorUpdateRoutine',
            description='DominantSensorUpdateRoutine',
        )
        # Dominant Sensor
        self.dominant_sensor_list = {}

    def update(self, data_list, cur_index, window_size):
        """
        Calculate the dominant sensor of current window and store
        the name of the sensor in the dominant sensor array. The
        information is fetched by dominant sensor features.
        :param data_list: list of sensor data
        :param cur_index: current data record index
        :param window_size: window size
        :return: None
        """
        sensor_count = {}
        for index in range(0, window_size):
            if data_list[cur_index - index]['sensor1'] in sensor_count.keys():
                sensor_count[data_list[cur_index - index]['sensor1']] += 1
            else:
                sensor_count[data_list[cur_index - index]['sensor1']] = 1
            if 'sensor2' in data_list[cur_index - index].keys():
                if data_list[cur_index - index]['sensor2'] in sensor_count.keys():
                    sensor_count[data_list[cur_index - index]['sensor2']] += 1
                else:
                    sensor_count[data_list[cur_index - index]['sensor2']] = 1
        # Find the Dominant one
        max_count = 0
        for sensor_label in sensor_count.keys():
            if sensor_count[sensor_label] > max_count:
                max_count = sensor_count[sensor_label]
                self.dominant_sensor_list[cur_index] = sensor_label

    def clear(self):
        """
        {inherit}
        """
        self.dominant_sensor_list = {}
