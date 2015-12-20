############################################################
# feature_calculation
# -------------------
# Use AlFeature and features defined in actlearn.feature to
# calculate statistical features with sliding window
############################################################

import os
import logging
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

if __name__ == '__main__':
    dataset_dir = '../datasets/bosch/'
    # Set current directory to local directory
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    # Loading Log configuration
    logging.config.fileConfig('../log/log.cfg')
    # Initialize AlData Structure
    data = AlData()
    # Load Translation File
    data.load_sensor_translation_from_file(dataset_dir + 'b1.translate')
    # Load Data File
    data.load_data_from_file(dataset_dir + 'b1')
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
    feature.add_feature(AlFeatureSensorCount(normalize=False))
    feature.add_feature(AlFeatureWindowDuration(normalize=False))
    feature.add_feature(AlFeatureEventHour(normalize=False))
    feature.add_feature(AlFeatureEventSensor(per_sensor=False))
    feature.add_feature(AlFeatureLastDominantSensor(per_sensor=False))
    feature.add_feature(AlFeatureEventSecond(normalize=False))
    feature.add_feature(AlFeatureSensorElapseTime(normalize=False))
    # Print Feature Summary
    feature.print_feature_summary()
    # Calculate Features
    feature.populate_feature_array(data.data)
    # Save to arff
    feature.save_data_as_arff('b1.arff')
    # Save to xls
    feature.save_data_as_xls('b1.xls', 0)


