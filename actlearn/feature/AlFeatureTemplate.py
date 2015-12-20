###########################################################
# AlFeatureTemplate
# ---------------------------------------------------------
# Template for features used to translate sensor data
# series into feature vector
###########################################################


class AlFeatureTemplate(object):

    def __init__(self, name, description, enabled=True, per_sensor=False, routine=None):
        """
        Initialization of Template Class
        :return:
        """
        # Name
        self.name = name
        # Description
        self.description = description
        # index
        self.index = -1
        # perSensor
        self.per_sensor = per_sensor
        # enable
        self.enabled = enabled
        # update Routine
        # For some feature, we will update statistical data every time we move forward
        # a data record. Instead of going back through previous window, the update function
        # in this routine structure will be called each time we advance to next data record
        self.routine = routine
        # sensor info
        self.sensorInfo = None

    def set_sensor_info(self, sensor_info):
        """
        Pass sensor info to feature
        :param sensor_info:
        :return:
        """
        self.sensorInfo = sensor_info
        if self.routine is not None:
            self.routine.set_sensor_info(sensor_info)

    def get_feature_value(self, data_list, cur_index, window_size, sensor_name=None):
        """
        Get Feature Value
        :param data_list: list of sensor data
        :param cur_index: current data record index
        :param window_size: window size
        :param sensor_name: name of sensor
        :return: a double value
        """
        return NotImplementedError()

    def is_value_valid(self):
        """
        :return: bool
        """
        return True
