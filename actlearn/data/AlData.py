import os
import sys
import logging
from actlearn.utils.get_date_time import *


class AlData(object):
    def __init__(self):
        """
        Initialize Class Variable
        :return: None
        """
        self.logger = logging.getLogger('AlData')
        # Sensor Dictionary: Sensor Information, Translation
        self.sensor_dict = {}
        # Activity Info: Activity Information
        self.activity_info = {}
        # Sensor Info: Sensor Information
        self.sensor_info = {}
        # Loaded Sensor Data
        self.data = []
        pass

    def load_sensor_translation_from_file(self, filename):
        """
        Load Sensor Translation from File
        :param filename: the path to the file containing sensor translation information
        :return: None
        """
        if os.path.isfile(filename):
            file_stream = open(filename, 'r')
            line_number = 0
            for line in file_stream:
                word_list = str(str(line).strip()).split()
                sensor_id = word_list[0]
                translation_labels = [word_list[1], word_list[2]]
                if sensor_id in self.sensor_dict.keys():
                    self.logger.error('repeated translation label definition found for sensor %s\n' % sensor_id)
                    self.logger.error('[current] %s:%d %s - %s %s' %
                                      (filename, line_number, sensor_id, word_list[1], word_list[2]))
                    self.logger.error('[previous] %s - %s %s' %
                                      (sensor_id, self.sensor_dict[sensor_id][0], self.sensor_dict[sensor_id][1]))
                else:
                    self.sensor_dict[sensor_id] = translation_labels
                line_number += 1
        else:
            self.logger.error('Cannot find sensor translation file %s\n' % filename)
        pass

    def load_data_from_file(self, filename, pre_translated=False, is_labeled=True):
        """
        Load Activity Training data from file
        :param filename: the path to the file containing sensor translation information
        :param pre_translated: the data file is translated or not
        :param is_labeled: the data file contain activity label or not
        :return: None
        """
        self.data = []
        if os.path.isfile(filename):
            file_stream = open(filename, 'r')
            line_number = 0
            for line in file_stream:
                line_number += 1
                word_list = str(str(line).strip()).split()
                cur_data_dict = {'datetime': get_date_time(word_list[0], word_list[1])}
                if pre_translated:
                    cur_data_dict['sensor1'] = word_list[2]
                    cur_data_dict['sensor2'] = word_list[3]
                    cur_data_dict['value'] = word_list[4]
                    if is_labeled:
                        cur_data_dict['activity'] = word_list[5]
                else:
                    sensor_id = word_list[2]
                    if len(self.sensor_dict) == 0:
                        # No translation available
                        cur_data_dict['sensor1'] = sensor_id
                        cur_data_dict['sensor2'] = None
                    elif sensor_id in self.sensor_dict.keys():
                        cur_data_dict['sensor1'] = self.sensor_dict[sensor_id][0]
                        cur_data_dict['sensor2'] = self.sensor_dict[sensor_id][1]
                    else:
                        self.logger.error('line %d: Cannot find sensor %s in sensor translation file' %
                                          (line_number, sensor_id))
                        return None
                    cur_data_dict['value'] = word_list[3]
                    if is_labeled:
                        cur_data_dict['activity'] = word_list[4]
                # Add Activity Frequency Calculation
                if is_labeled:
                    cur_activity_label = cur_data_dict['activity']
                    if cur_activity_label in self.activity_info.keys():
                        self.activity_info[cur_activity_label]['freq'] += 1
                    else:
                        self.activity_info[cur_activity_label] = {'freq': 1}
                # Add Sensor Frequency Calculation
                cur_sensor = cur_data_dict['sensor1']
                if cur_sensor is not None:
                    if cur_sensor in self.sensor_info.keys():
                        self.sensor_info[cur_sensor]['freq'] += 1
                    else:
                        self.sensor_info[cur_sensor] = {'freq': 1}
                cur_sensor = cur_data_dict['sensor2']
                if cur_sensor is not None:
                    if cur_sensor in self.sensor_info.keys():
                        self.sensor_info[cur_sensor]['freq'] += 1
                    else:
                        self.sensor_info[cur_sensor] = {'freq': 1}
                # Add curData to dataArray
                self.data.append(cur_data_dict)
        else:
            self.logger.error('Cannot find data file %s\n' % filename)

    def save_data_to_file(self, filename):
        '''
        Save loaded data array to file (with sensor id translated)
        :param filename: file name to store the translated data
        :return: None
        '''
        if self.data:
            f = open(filename, 'w')
            for cur_data_dict in self.data:
                f.write(cur_data_dict['datetime'].strftime('%Y-%m-%d %H:%M:%S.%f '))
                f.write('%s %s ' % (cur_data_dict['sensor1'], cur_data_dict['sensor2']))
                f.write('%s' % (cur_data_dict['value']))
                if 'activity' in cur_data_dict.keys():
                    f.write(' %s\n' % cur_data_dict['activity'])
                else:
                    f.write('\n')
            f.close()
        pass

    def calculate_window_size(self):
        """
        Based on Training Set, determine the optimal window
        size for each activity (one of 5, 10, 20, 30)
        :return: None
        """
        last_event = None
        act_count = 0
        act_window = {}
        # Go through all events and log how many events per activity
        for eventId in xrange(len(self.data)):
            event = self.data[eventId]
            cur_activity_label = event['activity']
            if last_event is not None:
                if last_event != cur_activity_label:
                    if last_event not in act_window.keys():
                        act_window[last_event] = {
                            '5': 0,
                            '10': 0,
                            '20': 0,
                            '30': 0
                        }
                    if act_count <= 5:
                        act_window[last_event]['5'] += 1
                    elif act_count <= 10:
                        act_window[last_event]['10'] += 1
                    elif act_count <= 20:
                        act_window[last_event]['20'] += 1
                    else:
                        act_window[last_event]['30'] += 1
                    act_count = 0
                else:
                    act_count += 1
            last_event = event['activity']
        # For each activity, find the window size with highest possibility
        for activityLabel in act_window.keys():
            max_tmp = 0
            for win_size in act_window[activityLabel].keys():
                if act_window[activityLabel][win_size] > max_tmp:
                    max_tmp = act_window[activityLabel][win_size]
                    self.activity_info[activityLabel]['winSize'] = int(win_size)

    def calculate_mostly_likely_activity_per_sensor(self):
        """
        Calculate mostly likely activity per sensor
        :return: None
        """
        sen_act = {}
        for event in self.data:
            cur_sensor_1 = event['sensor1']
            cur_sensor_2 = event['sensor2']
            cur_activity = event['activity']
            if cur_sensor_1 not in sen_act.keys():
                sen_act[cur_sensor_1] = {}
            if cur_sensor_2 not in sen_act.keys():
                sen_act[cur_sensor_2] = {}
            if cur_activity in sen_act[cur_sensor_1].keys():
                sen_act[cur_sensor_1][cur_activity] += 1
            else:
                sen_act[cur_sensor_1][cur_activity] = 1
            if cur_activity in sen_act[cur_sensor_2].keys():
                sen_act[cur_sensor_2][cur_activity] += 1
            else:
                sen_act[cur_sensor_2][cur_activity] = 1
        # Find the most likely activity
        for sensor in self.sensor_info.keys():
            max_tmp = 0
            for activity in sen_act[sensor].keys():
                if sen_act[sensor][activity] > max_tmp:
                    max_tmp = sen_act[sensor][activity]
                    self.sensor_info[sensor]['mostlikelyactivity'] = activity

    def print_data_summary(self, filename=None):
        """
        Print summary
        :param filename: file to write data. if None, write to stdout
        :return: None
        """
        if filename is None:
            f = sys.stdout
        else:
            f = open(filename, 'w')
        # Summary of activities, summary of sensors
        f.write('Activities: %d\n' % (len(self.activity_info)))
        f.write('\t')
        for activity in self.activity_info.keys():
            f.write('%s ' % activity)
        f.write('\n\n')
        f.write('Sensors: %d\n' % (len(self.sensor_info)))
        f.write('\t')
        for sensor in self.sensor_info.keys():
            f.write('%s ' % sensor)
        f.write('\n\n')

        # Print Summary for each activity
        for actLabel in self.activity_info.keys():
            f.write('%20s:\t' % actLabel)
            f.write('freq %10d\t\t' % self.activity_info[actLabel]['freq'])
            if 'winSize' in self.activity_info[actLabel].keys():
                f.write('winSize %d' % self.activity_info[actLabel]['winSize'])
            f.write('\n')

        # Print Summary for each translated sensor label
        f.write('\n')
        for sensorLabel in self.sensor_info.keys():
            f.write('%20s:\t' % sensorLabel)
            f.write('freq %10d\t\t' % self.sensor_info[sensorLabel]['freq'])
            if 'mostlikelyactivity' in self.sensor_info[sensorLabel].keys():
                f.write('activity %s' % self.sensor_info[sensorLabel]['mostlikelyactivity'])
            f.write('\n')
        f.write('\n')
