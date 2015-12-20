from .AlFeatureUpdateRoutineTemplate import AlFeatureUpdateRoutineTemplate


class AlFeatureUpdateSensorElapseTime (AlFeatureUpdateRoutineTemplate):

    def __init__(self):
        """
        Initialization of Template Class
        :return:
        """
        AlFeatureUpdateRoutineTemplate.__init__(
            self,
            name='SensorElapseTimeUpdateRoutine',
            description='Update Sensor Elapse Time for all enabled sensors',
        )
        # Sensor Fire Log
        self.sensor_fire_log = {}

    def update(self, data_list, cur_index, window_size):
        """
        Store the last fired time of each sensor
        :param data_list: list of sensor data
        :param cur_index: current data record index
        :param window_size: window size
        :return: None
        """
        if not self.sensor_fire_log:
            for sensor_label in self.sensor_info.keys():
                self.sensor_fire_log[sensor_label] = data_list[cur_index - window_size + 1]['datetime']
            for i in range(0, window_size):
                self.sensor_fire_log[data_list[cur_index - i]['sensor1']] = data_list[cur_index - i]['datetime']
                self.sensor_fire_log[data_list[cur_index - i]['sensor2']] = data_list[cur_index - i]['datetime']
        self.sensor_fire_log[data_list[cur_index]['sensor1']] = data_list[cur_index]['datetime']
        self.sensor_fire_log[data_list[cur_index]['sensor2']] = data_list[cur_index]['datetime']

    def clear(self):
        self.sensor_fire_log = {}

