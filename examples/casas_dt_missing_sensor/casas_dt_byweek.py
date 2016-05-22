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
    # Go through all bosch datasets
    datasets = ['b1', 'b2', 'b3']
    for datafile in datasets:
        feature_filename = 'feature_' + datafile + '.pkl'
        # Looking for processed feature data
        if os.path.exists(feature_filename):
            feature_file = open(feature_filename, mode='r')
            feature_dict = pickle.load(feature_file)
            feature = AlFeature()
            feature.load_from_dict(feature_dict)
        else:
            feature = load_casas_from_file(datafile, normalize=False, per_sensor=False)
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
            # For each sensor, if one is disabled throughout the week, get the result, and average results
            average_sensor_result = AlResult(result_name='%s dt missing sensor avg' % datafile,
                                             data_fname=datafile, mode='by_week')
            average_sensor_fname = 'dt_%s_missing_sensor_avg.pkl' % datafile
            if os.path.exists(average_sensor_fname):
                average_sensor_result.load_from_file(average_sensor_fname)
            if average_sensor_result.get_record_by_key(record_key) is None:
                num_sensors = 0
                avg_overall_performance = None
                avg_per_class_performance = None
                for sensor in feature.get_enabled_sensors():
                    num_sensors += 1
                    missing_sensor_result = AlResult(result_name='%s dt no sensor %s' % (datafile, sensor),
                                                     data_fname=datafile, mode='by_week')
                    missing_sensor_fname = 'dt_%s_missing_%s.pkl' % (datafile, sensor)
                    if os.path.exists(missing_sensor_fname):
                        missing_sensor_result.load_from_file(missing_sensor_fname)
                    # Check and see if current week is recorded or not
                    cur_record = missing_sensor_result.get_record_by_key(record_key)
                    if cur_record is None:
                        # Classify current data with missing sensor
                        mask_indices = feature.get_column_indices_by_sensor(sensor)
                        mask_array = np.ones((feature.num_enabled_features,))
                        mask_array[mask_indices] = 0
                        predicted_y = decision_tree.classify(feature.x[test_index][:] * mask_array)
                        confusion_matrix = get_confusion_matrix(num_classes=feature.num_enabled_activities,
                                                                label=feature.y[test_index], predicted=predicted_y)
                        (overall_performance, per_class_performance) = \
                            get_performance_array(num_classes=feature.num_enabled_activities,
                                                  confusion_matrix=confusion_matrix)
                        missing_sensor_result.add_record(decision_tree.export_to_dict(), key=record_key,
                                                         confusion_matrix=confusion_matrix,
                                                         overall_performance=overall_performance,
                                                         per_class_performance=per_class_performance)
                        missing_sensor_result.save_to_file(missing_sensor_fname)
                    if avg_overall_performance is None:
                        avg_overall_performance = missing_sensor_result.get_record_by_key(record_key)['overall_performance']
                    else:
                        avg_overall_performance += missing_sensor_result.get_record_by_key(record_key)['overall_performance']
                    if avg_per_class_performance is None:
                        avg_per_class_performance = missing_sensor_result.get_record_by_key(record_key)['per_class_performance']
                    else:
                        avg_per_class_performance += missing_sensor_result.get_record_by_key(record_key)['per_class_performance']
                average_sensor_result.add_record(decision_tree.export_to_dict(), key=record_key,
                                                 confusion_matrix=None,
                                                 overall_performance=avg_overall_performance/num_sensors,
                                                 per_class_performance=avg_per_class_performance/num_sensors)
                average_sensor_result.save_to_file(average_sensor_fname)
        learning_result.save_to_file(learning_result_fname)
