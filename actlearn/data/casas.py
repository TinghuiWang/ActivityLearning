import os
import logging.config
from actlearn.data.AlData import AlData
from actlearn.data.AlFeature import AlFeature
from actlearn.feature.lastEventHour import AlFeatureEventHour
from actlearn.feature.lastEventSeconds import AlFeatureEventSecond
from actlearn.feature.windowDuration import AlFeatureWindowDuration
from actlearn.feature.lastDominantSensor import AlFeatureLastDominantSensor
from actlearn.feature.lastSensorInWindow import AlFeatureEventSensor
from actlearn.feature.sensorCount import AlFeatureSensorCount
from actlearn.feature.sensorElapseTime import AlFeatureSensorElapseTime


def load_casas_from_file(data_filename, translation_filename=None, dataset_dir='../datasets/bosch/'):
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
    feature.add_feature(AlFeatureSensorCount(normalize=True))
    feature.add_feature(AlFeatureWindowDuration(normalize=True))
    feature.add_feature(AlFeatureEventHour(normalize=True))
    feature.add_feature(AlFeatureEventSensor(per_sensor=True))
    feature.add_feature(AlFeatureLastDominantSensor(per_sensor=True))
    feature.add_feature(AlFeatureEventSecond(normalize=True))
    feature.add_feature(AlFeatureSensorElapseTime(normalize=True))
    # Print Feature Summary
    feature.print_feature_summary()
    # Calculate Features
    feature.populate_feature_array(data.data)
    # Return features data
    return feature

