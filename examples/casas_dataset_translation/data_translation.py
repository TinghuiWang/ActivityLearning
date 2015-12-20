############################################################
# data_translation
# ----------------
# Simple usage of AlData structure to translate sensor IDs
# in original data log file into location abstract names.
############################################################

import os
import logging
import logging.config
from actlearn.data.AlData import AlData


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
    # Save annotated data to file
    data.save_data_to_file('b1.data')
    # Print out data summary
    data.print_data_summary()
