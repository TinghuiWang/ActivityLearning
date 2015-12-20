import sys
import logging
import math
import numpy as np


class AlFeature(object):
    """
    AlFeature class implements the interface of generating data for various
    training models.
    """

    def __init__(self):
        self.logger = logging.getLogger('AlFeature')
        self.max_window_size = 30
        # Activity List:
        # Dictionary of activity classes, indexed by their name.
        # including their index in target array, name, on/off switch
        self.activity_list = {}

        self.sensor_list = {}
        # enabledSensorCount:
        # Number of enabled sensors
        self.num_enabled_sensors = 0
        self.num_enabled_activities = 0
        self.num_enabled_features = 0
        self.num_static_features = 0
        self.num_per_sensor_features = 0

        self.feature_list = {}

        self.num_feature_windows = 1

        # Manage Update Data Routines
        self.routines = {}

        self.x = np.array([])
        self.y = np.array([])
        pass

    def add_activity(self, activity_label, window_size=30):
        """
        Add activity to activity_list with default value
        :param activity_label: activity name
        :param window_size: window size of current activity
        :return: None
        """
        if activity_label in self.activity_list.keys():
            self.logger.warn('activity: %s already existed. Add Activity Function ignored.' % activity_label)
        else:
            self.logger.debug('add activity class %s' % activity_label)
            self.activity_list[activity_label] = {}
            self.activity_list[activity_label]['name'] = activity_label
            self.activity_list[activity_label]['index'] = -1
            self.activity_list[activity_label]['enable'] = True
            self.activity_list[activity_label]['window_size'] = window_size
            self.assign_activity_indexes()

    def enable_activity(self, activity_label):
        """
        Enable certain Activity
        :param activity_label: name of activity
        :return: None
        """
        if activity_label in self.activity_list.keys():
            self.logger.info('Enable Activity %s' % activity_label)
            self.activity_list[activity_label]['enable'] = True
            self.assign_activity_indexes()
        else:
            self.logger.error('Activity %s not found' % activity_label)

    def disable_activity(self, activity_label):
        """
        Disable Specified activity
        :param activity_label: name of activity
        :return: None
        """
        if activity_label in self.activity_list.keys():
            self.logger.info('Disable Activity %s' % activity_label)
            self.activity_list[activity_label]['enable'] = False
            self.activity_list[activity_label]['index'] = -1
            self.assign_activity_indexes()
        else:
            self.logger.error('Activity %s not found' % activity_label)

    def assign_activity_indexes(self):
        """
        Assign index number to each activity enabled
        :return: Number of total enabled activity
        """
        i = 0
        for activity_label in self.activity_list.keys():
            activity = self.activity_list[activity_label]
            if activity['enable']:
                activity['index'] = i
                i += 1
            else:
                activity['index'] = -1
        self.num_enabled_activities = i
        self.logger.info('Finished assigning index to activities. %d Activities enabled' % i)
        return i

    def get_activities_by_indexes(self, activity_ids):
        """
        Get a group of activities by their corresponding indexes
        :param activity_ids:
        :return: An array of activity labels (array of string)
        """
        return [self.get_activities_by_indexes(cur_id) for cur_id in activity_ids]

    def get_activity_by_index(self, activity_id):
        """
        Get Activity name by their index
        :param activity_id: index number of activity
        :return: activity label (string)
        """
        for activity in self.activity_list:
            if activity_id == activity['index']:
                return activity

    def populate_activity_list(self, activity_info):
        """
        Populate Activity Configuration List by activity_info dictionary
        :param activity_info: dictionary containing activity information. Refer to AlData structure
        :return: None
        """
        for activity_label in activity_info.keys():
            if 'window_size' in activity_info[activity_label].keys():
                self.add_activity(activity_label, activity_info[activity_label]['window_size'])
            else:
                self.add_activity(activity_label)

    def add_sensor(self, sensor_label):
        """
        Add Sensor to sensorList and initialize configuration
        :param sensor_label: Name of sensor
        :return:
        """
        if sensor_label in self.sensor_list.keys():
            self.logger.warn('Sensor %s already imported. Add sensor operation will be ignored')
        else:
            self.logger.debug('Add sensor %s to sensor list' % sensor_label)
            self.sensor_list[sensor_label] = {}
            self.sensor_list[sensor_label]['name'] = sensor_label
            self.sensor_list[sensor_label]['index'] = -1
            self.sensor_list[sensor_label]['enable'] = True
            self.sensor_list[sensor_label]['lastFireTime'] = None
            self.assign_sensor_indexes()

    def enable_sensor(self, sensor_label):
        """
        Enable sensor by specifying the label
        :param sensor_label: sensor label
        :return:
        """
        if sensor_label in self.sensor_list.keys():
            self.logger.info('Enable Sensor %s' % sensor_label)
            self.sensor_list[sensor_label]['enable'] = True
            self.assign_sensor_indexes()
        else:
            self.logger.error('Failed to find sensor %s' % sensor_label)

    def disable_sensor(self, sensor_label):
        """
        Disable specified sensor in feature calculation
        :param sensor_label: sensor name
        :return:
        """
        if sensor_label in self.sensor_list.keys():
            self.logger.info('Disable Sensor %s' % sensor_label)
            self.sensor_list[sensor_label]['enable'] = False
            self.sensor_list[sensor_label]['index'] = -1
            self.assign_sensor_indexes()
        else:
            self.logger.error('Failed to find sensor %s' % sensor_label)

    def assign_sensor_indexes(self):
        """
        Assign Index to each enabled sensor
        :return: Number of enabled sensor
        """
        sensor_id = 0
        for sensor_label in self.sensor_list.keys():
            if self.sensor_list[sensor_label]['enable']:
                self.sensor_list[sensor_label]['index'] = sensor_id
                sensor_id += 1
            else:
                self.sensor_list[sensor_label]['index'] = -1
        self.num_enabled_sensors = sensor_id
        return sensor_id

    def get_sensor_by_index(self, index):
        """
        Get the name of sensor by index
        :param index: index of the sensor
        :return:
        """
        for sensor_name in self.sensor_list.keys():
            if self.sensor_list[sensor_name]['index'] == index:
                return sensor_name
        self.logger.error('Failed to find sensor with index %d' % index)
        return 'Not Found'

    def populate_sensor_list(self, sensor_info):
        """
        Populate Sensor Configuration List
        :param sensor_info:
        :return:
        """
        for sensor_label in sensor_info.keys():
            self.add_sensor(sensor_label)
        self.logger.info('Add %d Sensors into sensorList' % len(self.sensor_list))

    def add_routine(self, routine):
        """
        Add routine to feature update routine list
        :param routine:
        :return:
        """
        if routine.name in self.routines.keys():
            self.logger.info('feature routine %s already existed.' % routine.name)
        else:
            self.logger.debug('Add feature routine %s: %s' % (routine.name, routine.description))
            self.routines[routine.name] = routine

    def disable_routine(self, routine):
        """
        Check all enabled feature list and see if the routine is used by other features.
        If no feature need the routine, disable it
        :param routine:
        :return:
        """
        if routine.name in self.routines.keys():
            for feature_name in self.feature_list.keys():
                if self.feature_list[feature_name].enabled:
                    if self.feature_list[feature_name].routine == routine:
                        self.logger.debug('routine %s is used by feature %s.' % (routine.name, feature_name))
                        return
            self.logger.debug('routine %s is disabled.' % routine.name)
            self.routines[routine.name].enabled = False
        else:
            self.logger.error('routine %s not added to routine list' % routine.name)

    def enable_routine(self, routine):
        """
        Enable specified routine
        :param routine:
        :return:
        """
        if routine.name in self.routines.keys():
            self.logger.debug('routine %s is enabled.' % routine.name)
            routine.enabled = True
        else:
            self.logger.error('routine %s not added to routine list' % routine.name)

    def add_feature(self, feature):
        """
        Add Feature to featureList
        :param feature: AlFeatureTemplate Object
        :return:
        """
        if feature.name in self.feature_list.keys():
            self.logger.warn('feature: %s already existed. Add Feature Function ignored.' % feature.name)
        else:
            self.logger.debug('Add Feature %s: %s' % (feature.name, feature.description))
            self.feature_list[feature.name] = feature
            feature.set_sensor_info(self.sensor_list)
            if feature.routine is not None:
                self.add_routine(feature.routine)
            self.assign_feature_indexes()

    def disable_feature(self, feature_name):
        """
        Disable specific feature in featureList
        :param feature_name: name of feature
        :return: None
        """
        if feature_name in self.feature_list.keys():
            self.logger.info('Disable Feature %s: %s' % (feature_name, self.feature_list[feature_name]['description']))
            self.feature_list[feature_name].enabled = True
            self.feature_list[feature_name].index = -1
            self.assign_feature_indexes()
            if self.feature_list[feature_name].routine is not None:
                self.disable_routine(self.feature_list[feature_name].routine)
        else:
            self.logger.error('Feature %s Not Found' % feature_name)

    def enable_feature(self, feature_name):
        """
        Enable specific feature
        :param feature_name: Feature Name
        :return: None
        """
        if feature_name in self.feature_list.keys():
            self.logger.info('Enable Feature %s: %s' % (feature_name, self.feature_list[feature_name]['description']))
            self.feature_list[feature_name].enabled = True
            self.assign_feature_indexes()
            if self.feature_list[feature_name].routine is not None:
                self.enable_routine(self.feature_list[feature_name].routine)
        else:
            self.logger.error('Feature %s Not Found' % feature_name)

    def assign_feature_indexes(self):
        """
        Assign index to features
        :return: None
        """
        static_id = 0
        per_sensor_id = 0
        for featureLabel in self.feature_list.keys():
            feature = self.feature_list[featureLabel]
            if feature.enabled:
                if feature.per_sensor:
                    feature.index = per_sensor_id
                    per_sensor_id += 1
                else:
                    feature.index = static_id
                    static_id += 1
            else:
                feature.index = -1
        self.num_static_features = static_id
        self.num_per_sensor_features = per_sensor_id
        self.logger.info('Finished assigning index to features. %d Static Features, %d Per Sensor Features' %
                         (static_id, per_sensor_id))

    def count_feature_columns(self):
        """
        Count the size of feature columns
        :return: integer, size of feature columns
        """
        self.num_enabled_features = 0
        for feature_name in self.feature_list.keys():
            if self.feature_list[feature_name].enabled:
                if self.feature_list[feature_name].per_sensor:
                    self.num_enabled_features += self.num_enabled_sensors
                else:
                    self.num_enabled_features += 1
        return self.num_enabled_features * self.num_feature_windows

    def count_samples(self, data_list, is_labeled=True):
        """
        Count the maximum possible samples in data_list
        :param data_list: data list read by AlData module
        :param is_labeled: whether the data in the data_list is labeled?
        :return: integer
        """
        if len(data_list) < self.max_window_size - 1:
            self.logger.error('data size is %d smaller than window size %d' %
                              (len(data_list), self.max_window_size))
            return 0
        num_sample = 0
        if is_labeled:
            # If labeled, count enabled activity entry after the first
            # max_window_size event
            for dataEntry in data_list:
                if num_sample < self.max_window_size + self.num_feature_windows - 2:
                    num_sample += 1
                else:
                    """ ToDo: Need to check sensor enable status to make count sample count """
                    if self.activity_list[dataEntry['activity']]['enable']:
                        num_sample += 1
            num_sample -= self.max_window_size + self.num_feature_windows - 2
        else:
            # If not labeled, we need to calculate for each window
            # and finally find which catalog it belongs to
            num_sample = len(data_list) - self.max_window_size - self.num_feature_windows + 2
        return num_sample

    def populate_feature_array(self, data_list, is_labeled=True):
        """
        Populate Feature Array
        :param data_list: List of data
        :param is_labeled: If the data_list is labelled with activity label
        :return: None
        """
        num_feature_columns = self.count_feature_columns()
        num_feature_rows = self.count_samples(data_list, is_labeled)
        self.x = np.zeros((num_feature_rows, num_feature_columns), dtype=np.float)
        self.y = np.zeros(num_feature_rows, dtype=np.integer)
        cur_row_id = self.max_window_size - 1
        cur_sample_id = 0
        # Execute feature update routine
        for (key, routine) in self.routines.items():
            if routine.enabled:
                routine.clear()
        while cur_row_id < len(data_list):
            cur_sample_id += self.calculate_window_feature(data_list, cur_row_id, cur_sample_id, is_labeled)
            cur_row_id += 1
        # Due to sensor event discontinuity, the sample size will be smaller than the num_feature_rows calculated
        self.x = self.x[0:cur_sample_id, :]
        self.logger.info('Total amount of feature vectors calculated: %d' % cur_sample_id)

    def calculate_window_feature(self, data_list, cur_row_id, cur_sample_id, is_labeled=True):
        """
        Calculate feature vector for current window specified
        by cur_row_id (> self.max_window_size)
        :param data_list: List of data
        :param cur_row_id: row index of current window (last row)
        :param cur_sample_id: Row Index of current sample in self.x
        :param is_labeled: If the data_list is labelled with activity label
        :return: 1 - feature added to array, 0 - window skipped
        """
        # Default Window Size to 30
        window_size = self.max_window_size
        # Skip current window if labeled activity is ignored
        if is_labeled:
            activity_label = data_list[cur_row_id]['activity']
            window_size = self.activity_list[activity_label]['window_size']
            if not self.activity_list[activity_label]['enable']:
                return 0
        if cur_row_id > self.max_window_size - 1:
            if cur_sample_id == 0:
                for i in range(self.num_enabled_features * (self.num_feature_windows - 1)):
                    self.x[cur_sample_id][self.num_enabled_features*self.num_feature_windows-i-1] = \
                        self.x[cur_sample_id][self.num_enabled_features * (self.num_feature_windows-1)-i-1]
            else:
                for i in range(self.num_enabled_features * (self.num_feature_windows - 1)):
                    self.x[cur_sample_id][self.num_enabled_features*self.num_feature_windows-i-1] = \
                        self.x[cur_sample_id-1][self.num_enabled_features * (self.num_feature_windows-1)-i-1]
        # Execute feature update routine
        for (key, routine) in self.routines.items():
            if routine.enabled:
                routine.update(data_list, cur_row_id, window_size)
        # Get Feature Data and Put into arFeature array
        for (key, feature) in self.feature_list.items():
            if feature.enabled:
                # If it is per Sensor index, we need to iterate through all sensors to calculate
                if feature.per_sensor:
                    for sensor_label in self.sensor_list.keys():
                        if self.sensor_list[sensor_label]['enable']:
                            column_index = self.num_static_features + \
                                           feature.index * self.num_enabled_sensors + \
                                           self.sensor_list[sensor_label]['index']
                            self.x[cur_sample_id][column_index] = \
                                feature.get_feature_value(data_list, cur_row_id, window_size, sensor_label)
                else:
                    self.x[cur_sample_id][feature.index] = \
                        feature.get_feature_value(data_list, cur_row_id, window_size)
                if not feature.is_value_valid():
                    return 0
        if cur_row_id < self.max_window_size + self.num_feature_windows - 2:
            return 0
        if is_labeled:
            self.y[cur_sample_id] = self.activity_list[data_list[cur_row_id]['activity']]['index']
        return 1

    def print_feature_summary(self):
        """
        Print Feature Status
        :return:
        """
        for feature_name in self.feature_list.keys():
            sys.stdout.write('%25s:\t' % feature_name)
            feature = self.feature_list[feature_name]
            if feature.enabled:
                sys.stdout.write('Enabled\t')
            else:
                sys.stdout.write('Disabled\t')
            sys.stdout.write('%5d\t' % feature.index)
            sys.stdout.write('%s\n' % feature.description)

    def get_feature_by_index(self, index):
        """
        Get Feature Name by Index
        :param index: Index of feature
        :return: (feature label, sensor label) tuple.
                 If it is not a per_sensor feature, sensor label is None
        """
        max_id = self.num_enabled_features
        if index > max_id:
            self.logger.error('index %d is greater than the number of feature columns %d' %
                              (index, max_id))
        if index > self.num_static_features:
            # It is per_sensor Feature
            sensor_id = (index - self.num_static_features) % self.num_enabled_sensors
            feature_id = math.floor((index - self.num_static_features) / self.num_enabled_sensors)
            per_sensor = True
        else:
            # It is a generic feature
            sensor_id = -1
            feature_id = index
            per_sensor = False
        # Find Corresponding feature name and sensor label
        feature_name = None
        for featureLabel in self.feature_list.keys():
            feature = self.feature_list[featureLabel]
            if feature.index == feature_id and feature.per_sensor == per_sensor:
                feature_name = featureLabel
                break
        sensor_name = None
        if sensor_id > 0:
            for sensor_label in self.sensor_list.keys():
                sensor = self.sensor_list[sensor_label]
                if sensor['index'] == sensor_id:
                    sensor_name = sensor_label
                    break
        return feature_name, sensor_name

    def save_data_as_arff(self, filename=None):
        """
        Save populated feature data array as ARFF file
        :param filename: Name of arff file
        :return:
        """
        if filename is None:
            f = sys.stdout
        else:
            f = open(filename, 'w')
        f.write('@relation ar\n\n')
        # Populate Feature Lists
        for index in range(0, self.count_feature_columns()):
            (feature_name, sensor_label) = self.get_feature_by_index(index % self.num_enabled_features)
            if sensor_label:
                f.write('@attribute %s-%s numeric\n' %
                        (feature_name, sensor_label))
            else:
                f.write('@attribute %s numeric\n' % feature_name)
        # Populate Class Target
        f.write('@attribute class {')
        for index in range(0, self.num_enabled_activities):
            if index == self.num_enabled_activities - 1:
                f.write('%d}' % index)
            else:
                f.write('%d,' % index)
        f.write('\n\n')
        # Populate Data
        f.write('@data\n')
        num_rows, num_cols = self.x.shape
        for row_id in range(0, num_rows):
            for col_id in range(0, num_cols):
                if col_id == num_cols - 1:
                    f.write('%f\n' % self.x[row_id][col_id])
                else:
                    f.write('%f,' % self.x[row_id][col_id])
            f.write('%d' % self.y[row_id])
        f.close()

    def save_data_as_xls(self, filename, start_id):
        """
        Save partial data into xls format (65530 maximum entries)
        :param filename: name of the xls file
        :param start_id: the start position to log the data
        :return:
        """
        import xlwt
        book = xlwt.Workbook()
        # Feature Description Sheet
        feature_sheet = book.add_sheet('Features')
        feature_list_title = ['name', 'index', 'enabled', 'per_sensor', 'description', 'routine']
        for c in range(0, len(feature_list_title)):
            feature_sheet.write(0, c, str(feature_list_title[c]))
        r = 1
        for feature in self.feature_list:
            feature_sheet.write(r, 0, str(self.feature_list[feature].name))
            feature_sheet.write(r, 1, str(self.feature_list[feature].index))
            feature_sheet.write(r, 2, str(self.feature_list[feature].enabled))
            feature_sheet.write(r, 3, str(self.feature_list[feature].per_sensor))
            feature_sheet.write(r, 4, str(self.feature_list[feature].description))
            if self.feature_list[feature].routine is None:
                feature_sheet.write(r, 5, 'None')
            else:
                feature_sheet.write(r, 5, str(self.feature_list[feature].routine.name))
            r += 1
        # Activity Information Sheet
        activity_sheet = book.add_sheet('Activity')
        c = 0
        for item in self.activity_list.items()[0][1].keys():
            activity_sheet.write(0, c, str(item))
            c += 1
        r = 1
        for activity in self.activity_list.keys():
            c = 0
            for item in self.activity_list[activity].keys():
                activity_sheet.write(r, c, str(self.activity_list[activity][item]))
                c += 1
            r += 1
        # Sensor Information Sheet
        sensor_sheet = book.add_sheet('Sensor')
        c = 0
        for item in self.sensor_list.items()[0][1].keys():
            sensor_sheet.write(0, c, str(item))
            c += 1
        r = 1
        for sensor in self.sensor_list.keys():
            c = 0
            for item in self.sensor_list[sensor].keys():
                sensor_sheet.write(r, c, str(self.sensor_list[sensor][item]))
                c += 1
            r += 1
        # Data Sheet
        data_sheet = book.add_sheet('Data')
        data_sheet.write(0, 0, 'activity')
        # Calculate enabled sensor size
        num_sensors = self.num_enabled_sensors
        # Add Feature Title
        for feature_name in self.feature_list.keys():
            if self.feature_list[feature_name].enabled:
                if self.feature_list[feature_name].per_sensor:
                    # Calculate Start Position
                    start_col = self.num_static_features + self.feature_list[feature_name].index * num_sensors + 1
                    data_sheet.write_merge(0, 0, start_col, start_col + num_sensors - 1, feature_name)
                else:
                    data_sheet.write(0, self.feature_list[feature_name].index + 1, feature_name)
        for c in range(1, self.num_static_features + 1):
            data_sheet.write(1, c, 'window')
        for f in range(0, self.num_per_sensor_features):
            for sensor in self.sensor_list.keys():
                start_col = f*num_sensors + self.num_static_features + self.sensor_list[sensor]['index'] + 1
                data_sheet.write(1, start_col, sensor)
        # Add Data from Data Array
        r = 2
        (num_samples, num_features) = self.x.shape
        if num_samples - start_id > 65530:
            end_id = start_id + 65530
        else:
            end_id = num_samples
        for i in range(start_id, end_id):
            data_sheet.write(r, 0, str(self.y[i]))
            c = 1
            for item in self.x[i]:
                data_sheet.write(r, c, str(item))
                c += 1
            r += 1
        # Save Workbook
        book.save(filename)
        pass
