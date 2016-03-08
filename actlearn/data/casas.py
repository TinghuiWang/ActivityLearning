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


def get_week_boundary(feature):
    """
    Indices boudary separated by weeks
    :type feature: AlFeature
    :param feature:
    :return: array
    """
    week_array = []
    last_record = datetime.fromtimestamp(feature.time[0]).date()
    for i in range(feature.time.shape[0]):
        today = datetime.fromtimestamp(feature.time[i]).date()
        delta = today - last_record
        if delta.days >= 7:
            last_record = today
            week_array.append(i)
    return week_array


def plot_casas_learning_curve(filename_array):
    """
    Plot Learning Curve for CASAS data on a day-by-day or week-by-week bases
    :type filename_array: list of str
    :param filename_array:
    :return:
    """
    logger = actlearn_logger.get_logger('casas_learning_curve')
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
    # Check the base of all learning curve files
    # 1. Same length?
    # 2. By week or by day?
    week_or_day = learning_data[0]['week_or_day']
    curve_length = len(learning_data[0]['value'])
    for curve in learning_data:
        if curve['week_or_day'] != week_or_day:
            load_error = True
        cur_length = len(curve['value'])
        if cur_length != curve_length:
            logger.warning('In consistence length of learning curve: %s has %d items' %
                           (curve['name'], cur_length))
            if cur_length < curve_length:
                curve_length = cur_length
    if load_error:
        logger.error('Error in step size of learning curve')
        for curve in learning_data:
            logger.error('%20s: by %s' % (curve['name'], curve['week_or_day']))
        return
    # Plot it using plt
    x = range(curve_length)
    x_label = ['week %d' % (i + 1) for i in x]
    y = range(0, 100, 10)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for curve in learning_data:
        plt.plot(x, curve['value'], label=curve['name'])
    ax.set_xticks(x)
    ax.set_xticklabels(x_label, rotation='vertical')
    ax.set_yticks(y)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels)
    plt.show()
    return


def save_casas_learning_curve(filename, curve_name, value_array, week_or_day):
    """
    Save Learning Curve to file
    :param filename:
    :param curve_name:
    :param value_array:
    :type week_or_day: str
    :param week_or_day: by week or by day
    :return:
    """
    fp = open(filename, 'w+')
    data = {
        'name': curve_name,
        'value': value_array,
        'week_or_day': week_or_day
    }
    pickle.dump(data, fp, protocol=-1)
    fp.close()
