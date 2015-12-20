###########################################################
# AlFeatureUpdateRoutineTemplate
# ---------------------------------------------------------
# Template for feature update routine structure
###########################################################


class AlFeatureUpdateRoutineTemplate(object):

    def __init__(self, name, description, enabled=True):
        """
        Initialization of Template Class
        :return:
        """
        # Name
        self.name = name
        # Description
        self.description = description
        # enable
        self.enabled = enabled
        # sensor info
        self.sensor_info = None

    def set_sensor_info(self, sensor_info):
        """
        Pass sensor info to routine
        :param sensor_info: sensor info structure
        :return: None
        """
        self.sensor_info = sensor_info

    def update(self, data_list, cur_index, window_size):
        """
        For some features, we will update some statistical data every time
        we move forward a data record, instead of going back through the whole
        window and try to find the answer. This function will be called every time
        we advance in data record.
        :param data_list: list of sensor data
        :param cur_index: current data record index
        :param window_size: window size
        :return: None
        """
        return NotImplementedError()

    def clear(self):
        """
        Clear Internal Data Structures
        :return: None
        """
        pass