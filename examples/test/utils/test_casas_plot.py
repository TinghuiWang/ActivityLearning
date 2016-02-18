import os
import cPickle as pickle
import numpy as np
from actlearn.data.AlFeature import AlFeature
from actlearn.data.casas import load_casas_from_file
from actlearn.utils.event_bar_plot import event_bar_plot


if __name__ == '__main__':
    # Set current directory to local directory
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    # Go through all bosch datasets
    datasets = ['b1']
    for datafile in datasets:
        feature_filename = 'feature_' + datafile + '.pkl'
        # Looking for processed feature data
        if os.path.exists(feature_filename):
            feature_file = open(feature_filename, mode='r')
            feature_dict = pickle.load(feature_file)
            feature = AlFeature()
            feature.load_from_dict(feature_dict)
        else:
            feature = load_casas_from_file(datafile, datafile + '.translate', dataset_dir='../../datasets/bosch/')
            feature_file = open(feature_filename, mode='w')
            pickle.dump(feature.export_to_dict(), feature_file, protocol=-1)
        feature_file.close()
        # feature.save_data_as_xls('tmp.xls', 0)
        # event_bar_plot(feature.time[0:10000], feature.y[0:10000], feature.num_enabled_activities,
        #                classified=feature.y[1:10001], ignore_activity=feature.activity_list['Other_Activity']['index'])
        event_bar_plot(feature.time[0:100000], feature.y[0:100000], feature.num_enabled_activities,
                       ignore_activity=feature.activity_list['Other_Activity']['index'])
