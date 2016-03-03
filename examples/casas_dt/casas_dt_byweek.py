import os
import sys
import cPickle as pickle
import numpy as np
import pydot
import xlwt
from actlearn.data.casas import load_casas_from_file, get_week_boundary, save_casas_learning_curve, plot_casas_learning_curve
from actlearn.decision_tree.tree import DecisionTree
from actlearn.utils.confusion_matrix import get_confusion_matrix
from actlearn.utils.classifier_performance import performance_index, get_performance_array
from actlearn.data.AlFeature import AlFeature
from actlearn.utils.event_bar_plot import event_bar_plot

if __name__ == '__main__':
    # Set current directory to local directory
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    # Setup Worksheet to store learning result
    book = xlwt.Workbook()
    overall_sheet = book.add_sheet('overall')
    overall_list_title = ['dataset', 'correctness']
    overall_list_row = 0
    for c in range(len(overall_list_title)):
        overall_sheet.write(0, c, str(overall_list_title[c]))
    dataset_list_title = ['activities'] + performance_index
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
            feature = load_casas_from_file(datafile, datafile + '.translate', normalize=False, per_sensor=False)
            feature_file = open(feature_filename, mode='w')
            pickle.dump(feature.export_to_dict(), feature_file, protocol=-1)
        feature_file.close()
        num_samples = feature.x.shape[0]
        train_index = []
        test_index = []
        week_array = get_week_boundary(feature)
        learning_result = []
        for week_id in range(len(week_array) - 1):
            train_index = range(0, week_array[week_id])
            test_index = range(week_array[week_id], week_array[week_id + 1])
            decision_tree = DecisionTree(feature.x.shape[1], feature.num_enabled_activities)
            # Load Decision Tree Data
            dt_filename = 'dt_' + datafile + ('_week%3d.pkl' % week_id)
            if os.path.exists(dt_filename):
                dt_file = open(dt_filename, mode='r')
                dt_dict = pickle.load(dt_file)
                decision_tree.load_from_dict(dt_dict)
            else:
                decision_tree.build(feature.x[train_index][:], feature.y[train_index])
                dt_dict = decision_tree.export_to_dict()
                dt_file = open(dt_filename, mode='w')
                pickle.dump(dt_dict, dt_file, protocol=-1)
            dt_file.close()
            # print(feature.x.shape[0])
            predicted_y = decision_tree.classify(feature.x[test_index][:])
            # print(predicted_y)
            # event_bar_plot(feature.time[test_index], feature.y[test_index], feature.num_enabled_activities,
            #                classified=np.asarray(predicted_y, dtype=np.int),
            #                ignore_activity=feature.activity_list['Other_Activity']['index'],
            #                max_days=10)
            confusion_matrix = get_confusion_matrix(num_classes=feature.num_enabled_activities,
                                                    label=feature.y[test_index], predicted=predicted_y)
            correctness = (confusion_matrix.trace() / float(predicted_y.shape[0])) * 100
            learning_result.append(correctness)
            training_stat = get_performance_array(num_classes=feature.num_enabled_activities,
                                                  confusion_matrix=confusion_matrix)
            overall_list_row += 1
            overall_sheet.write(overall_list_row, 0, datafile)
            overall_sheet.write(overall_list_row, 1, '%.5f' % correctness)
            overall_sheet.write(overall_list_row, 2, ('week %3d' % week_id))
            dataset_sheet = book.add_sheet(datafile + (' week %3d' % week_id))
            for c in range(0, len(dataset_list_title)):
                dataset_sheet.write(0, c, str(dataset_list_title[c]))
            num_performance = len(performance_index)
            for r in range(feature.num_enabled_activities):
                activity_label = feature.get_activity_by_index(r)
                dataset_sheet.write(r+1, 0, activity_label)
                for c in range(num_performance):
                    dataset_sheet.write(r+1, c+1, '%.5f' % training_stat[r][c])
        learning_curve_fname = 'dt_learning_' + datafile + '.pkl'
        save_casas_learning_curve(learning_curve_fname, '%s decision tree' % datafile, learning_result, 'week')
    book.save('casas_dt_results.xls')
    plot_casas_learning_curve(['dt_learning_b1.pkl'])
