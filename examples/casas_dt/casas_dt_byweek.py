import os
import cPickle as pickle
import xlwt
from actlearn.data.casas import load_casas_from_file, get_boundary
from actlearn.decision_tree.tree import DecisionTree
from actlearn.utils.confusion_matrix import get_confusion_matrix
from actlearn.utils.classifier_performance import *
from actlearn.data.AlFeature import AlFeature
from actlearn.utils.event_bar_plot import event_bar_plot
from actlearn.utils.AlResult import AlResult

if __name__ == '__main__':
    # Set current directory to local directory
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    # Setup Worksheet to store learning result
    book = xlwt.Workbook()
    overall_sheet = book.add_sheet('overall')
    overall_list_title = ['dataset', '#week'] + overall_performance_index
    overall_list_row = 0
    for c in range(len(overall_list_title)):
        overall_sheet.write(0, c, str(overall_list_title[c]))
    dataset_list_title = ['activities'] + per_class_performance_index
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
        week_array = get_boundary(feature, period='week')
        learning_result_fname = 'dt_learning_' + datafile + '.pkl'
        learning_result = AlResult(result_name='%s decision tree' % datafile, data_fname=datafile, mode='by_week')
        if os.path.exists(learning_result_fname):
            learning_result.load_from_file(learning_result_fname)
        for week_id in range(len(week_array) - 1):
            train_index = range(0, week_array[week_id])
            test_index = range(week_array[week_id], week_array[week_id + 1])
            decision_tree = DecisionTree(feature.x.shape[1], feature.num_enabled_activities)
            # Load Decision Tree Data
            record_key = 'week %d' % week_id
            record = learning_result.get_record_by_key(record_key)
            if record is None:
                decision_tree.build(feature.x[train_index][:], feature.y[train_index])
                predicted_y = decision_tree.classify(feature.x[test_index][:])
                confusion_matrix = get_confusion_matrix(num_classes=feature.num_enabled_activities,
                                                        label=feature.y[test_index], predicted=predicted_y)
                (overall_performance, per_class_performance) = \
                    get_performance_array(num_classes=feature.num_enabled_activities,
                                          confusion_matrix=confusion_matrix)
                learning_result.add_record(decision_tree.export_to_dict(), key=record_key,
                                           confusion_matrix=confusion_matrix,
                                           overall_performance=overall_performance,
                                           per_class_performance=per_class_performance)
            else:
                decision_tree.load_from_dict(record['model'])
                confusion_matrix = record['confusion_matrix']
                overall_performance = record['overall_performance']
                per_class_performance = record['per_class_performance']
            overall_list_row += 1
            overall_sheet.write(overall_list_row, 0, datafile)
            overall_sheet.write(overall_list_row, 1, ('week %3d' % week_id))
            for c in range(len(overall_performance_index)):
                overall_sheet.write(overall_list_row, c + 2, '%.5f' % overall_performance[c])
            dataset_sheet = book.add_sheet(datafile + (' week %3d' % week_id))
            for c in range(0, len(dataset_list_title)):
                dataset_sheet.write(0, c, str(dataset_list_title[c]))
            num_performance = len(per_class_performance_index)
            for r in range(feature.num_enabled_activities):
                activity_label = feature.get_activity_by_index(r)
                dataset_sheet.write(r+1, 0, activity_label)
                for c in range(num_performance):
                    dataset_sheet.write(r+1, c+1, per_class_performance[r][c])
        learning_result.save_to_file(learning_result_fname)
    book.save('casas_dt_results.xls')
