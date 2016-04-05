import os
import logging.config
import cPickle as pickle
from datetime import datetime
import matplotlib.pyplot as plt
from actlearn.log.logger import actlearn_logger
from actlearn.data.AlData import AlData
from actlearn.data.AlFeature import AlFeature
from actlearn.feature.lastEventHour import AlFeatureEventHour
from actlearn.feature.lastEventSeconds import AlFeatureEventSecond
from actlearn.feature.windowDuration import AlFeatureWindowDuration
from actlearn.feature.lastDominantSensor import AlFeatureLastDominantSensor
from actlearn.feature.lastSensorInWindow import AlFeatureEventSensor
from actlearn.feature.sensorCount import AlFeatureSensorCount
from actlearn.feature.sensorElapseTime import AlFeatureSensorElapseTime


def load_casas_from_file(data_filename, translation_filename=None,
                         dataset_dir='../datasets/bosch/',
                         normalize=True, per_sensor=True, ignore_other=False):
    """
    Load CASAS Data From File
    :param data_filename:
    :param translation_filename:
    :param dataset_dir:
    :param normalize:
    :param per_sensor:
    :param ignore_other:
    :return:
    """
    # Initialize AlData Structure
    data = AlData()
    # Load Translation File
    if translation_filename is not None:
        data.load_sensor_translation_from_file(dataset_dir + translation_filename)
    # Load Data File
    data.load_data_from_file(dataset_dir + data_filename)
    # Some basic statistical calculations
    data.calculate_window_size()
    data.calculate_mostly_likely_activity_per_sensor()
    # Print out data summary
    data.print_data_summary()
    # Configure Features
    feature = AlFeature()
    # Pass Activity and Sensor Info to AlFeature
    feature.populate_activity_list(data.activity_info)
    feature.populate_sensor_list(data.sensor_info)
    # feature.DisableActivity('Other_Activity')
    # Add lastEventHour Feature
    feature.featureWindowNum = 1
    feature.add_feature(AlFeatureSensorCount(normalize=normalize))
    feature.add_feature(AlFeatureWindowDuration(normalize=normalize))
    feature.add_feature(AlFeatureEventHour(normalize=normalize))
    feature.add_feature(AlFeatureEventSensor(per_sensor=per_sensor))
    feature.add_feature(AlFeatureLastDominantSensor(per_sensor=per_sensor))
    feature.add_feature(AlFeatureEventSecond(normalize=normalize))
    feature.add_feature(AlFeatureSensorElapseTime(normalize=normalize))
    # Select whether disable other activity or not
    if ignore_other:
        feature.disable_activity('Other_Activity')
    # Print Feature Summary
    feature.print_feature_summary()
    # Calculate Features
    feature.populate_feature_array(data.data)
    # Return features data
    return feature


def get_boundary(feature, period='week'):
    """
    Indices boudary separated by weeks
    :type feature: AlFeature
    :param feature:
    :type period: str
    :param period: week or day (week - 7, day - 1, month - 31)
    :return: array
    """
    if period == 'month':
        period_in_days = 30
    elif period == 'day':
        period_in_days = 1
    else:
        period_in_days = 7
    week_array = []
    last_record = datetime.fromtimestamp(feature.time[0]).date()
    for i in range(feature.time.shape[0]):
        today = datetime.fromtimestamp(feature.time[i]).date()
        delta = today - last_record
        if delta.days >= period_in_days:
            last_record = today
            week_array.append(i)
    return week_array


def casas_where_are_the_errors(y, predicted_y, label_array, filename=None):
    """
    Go through the prediction and see how errors occur.
    The function classify them into three categories:
    0. The errors that can be fixed by enforcing activity continuity
    1. Shifted Start point of activity
    2. Shifted End point of activity
    3. Errors that totally misclassified an activity
    4. Other mistakes
    :type y: numpy.array
    :param y: original labels
    :type predicted_y: numpy.array
    :param predicted_y: predicted labels
    :type label_array: list
    :param label_array: list of labels organized in order
    :type filename: str
    :param filename: Name of the pkl file to which summary is saved
    :return: dictionary that contain all the information
    """
    error_summary = {}
    for label in label_array:
        error_summary[label] = {
            'glitch': 0.,
            'shifted_in': 0.,
            'shifted_out': 0.,
            'misclassification': 0.,
            'other': 0.
        }
    prev_label = y[0]
    start = 0
    total_error = 0
    for i in range(y.shape[0]):
        cur_label = y[i]
        # Record Total Errors
        if y[i] != predicted_y[i]:
            total_error += 1
        # If an activity comes to an end
        if cur_label != prev_label:
            end = i
            # Look for Offset Error
            error_status = 'shifted_in'
            offset_label = -1
            label_right = 0
            miss = 0.
            glitch = 0.
            # Check if it is a total miss
            for j in range(start, end):
                # Find errors and then determine what type it is
                if predicted_y[j] != y[j]:
                    if label_right == 0 and (offset_label == -1 or predicted_y[j] == offset_label):
                        # Offset In Calculation
                        offset_label = predicted_y[j]
                        miss += 1
                        error_status = 'shifted_in'
                    elif label_right == 1 and (offset_label == -1 or predicted_y[j] == offset_label):
                        # If a new label (other than previous shifted label),
                        # and we did get new label classified, count it as glitch
                        offset_label = predicted_y[j]
                        miss += 1
                        error_status = 'glitch'
                    else:
                        # We have not got the right label, and the classification changed to other label
                        # it is still a miss and submit it to other
                        offset_label = -2
                        error_status = 'other'
                        miss += 1
                else:
                    # We arrive at a point where they are the same
                    # Submit calculated offset and stop counting offset
                    error_summary[label_array[y[j]]][error_status] += miss
                    miss = 0
                    offset_label = -1
                    label_right = 1
            if miss == (end - start):
                # It is a total misclassification
                error_summary[label_array[prev_label]]['misclassification'] += miss
            else:
                # It is shift out instead of glitch
                error_summary[label_array[prev_label]]['shifted_out'] += miss
            # Update Start, End, and label
            start = i
            prev_label = cur_label
    if filename is not None:
        fp = open(filename, 'w+')
        pickle.dump(error_summary, fp, protocol=-1)
        fp.close()
    return error_summary


def plot_error_summary(filename_array, label_array):
    """
    :return:
    """
    logger = actlearn_logger.get_logger('casas_error_summary')
    # Load data into array of dictionary
    load_error = False
    learning_data = []
    for fname in filename_array:
        if os.path.exists(fname):
            fp = open(fname, 'r')
            learning_data.append(pickle.load(fp))
        else:
            logger.error('Cannot find file %s' % fname)
            load_error = True
    if load_error:
        return
