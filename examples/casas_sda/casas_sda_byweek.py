import os
import sys
import time
import theano
import theano.tensor as T
import cPickle as pickle
import numpy as np
import xlwt
import pydot
from actlearn.data.casas import load_casas_from_file, get_week_boundary, save_casas_learning_curve, plot_casas_learning_curve
from actlearn.training_algorithms.cross_validation import run_cross_validation
from actlearn.models.StackedDenoisingAutoencoder import StackedDenoisingAutoencoder
from actlearn.utils.confusion_matrix import get_confusion_matrix
from actlearn.utils.classifier_performance import performance_index, get_performance_array
from actlearn.data.AlFeature import AlFeature
from actlearn.utils.event_bar_plot import event_bar_plot


if __name__ == '__main__':
    # Set current directory to local directory
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    book = xlwt.Workbook()
    overall_sheet = book.add_sheet('overall')
    overall_list_title = ['dataset', 'hidden units', 'correctness']
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
            feature = load_casas_from_file(datafile, datafile + '.translate', normalize=True, per_sensor=True,
                                           ignore_other=False)
            feature_file = open(feature_filename, mode='w')
            pickle.dump(feature.export_to_dict(), feature_file, protocol=-1)
        feature_file.close()
        num_samples = feature.x.shape[0]
        train_index = []
        test_index = []
        x_tensor = theano.shared(np.asarray(feature.x, dtype=theano.config.floatX), borrow=True)
        y_tensor = T.cast(theano.shared(feature.y, borrow=True), 'int32')
        week_array = get_week_boundary(feature)
        # Number of perceptrons in hidden layer
        hidden_layer_list = [[200, 200, 200]]
        for hidden_layer in hidden_layer_list:
            input_x = T.matrix('x')
            num_classes = feature.num_enabled_activities
            numpy_rng = np.random.RandomState(int(time.clock()))
            model = StackedDenoisingAutoencoder(numpy_rng=numpy_rng, input=input_x,
                                                n_ins=feature.x.shape[1], n_outs=num_classes,
                                                hidden_layers_sizes=hidden_layer,
                                                corruption_levels=[0, 0, 0])
            learning_result = []
            for week_id in range(len(week_array) - 1):
                if week_id == 0:
                    train_index = range(0, week_array[week_id])
                else:
                    train_index = range(week_array[week_id - 1], week_array[week_id])
                test_index = range(week_array[week_id], week_array[week_id + 1])
                # Load MLP if exists
                sda_filename = datafile + '_' + \
                    '_'.join([('%d' % hidden_layer[layer]) for layer in range(len(hidden_layer))]) + \
                    ('_week%03d.pkl' % week_id)
                if os.path.exists(sda_filename):
                    # Load Model from File
                    model.load(sda_filename)
                else:
                    model.do_pretraining(data=x_tensor[train_index],
                                         num_data=len(train_index), batch_size=10,
                                         learning_rate_array=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                                         num_epochs=15)
                    model.do_fine_tuning(data=x_tensor[train_index], label=y_tensor[train_index],
                                         num_data=len(train_index), batch_size=10,
                                         learning_rate_array=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                                         num_epochs=10)
                    model.save(sda_filename)
                # Performance Evaluation
                result = model.classify(x_tensor[test_index])
                predicted_y = result[0]
                confusion_matrix = get_confusion_matrix(num_classes=feature.num_enabled_activities,
                                                        label=feature.y[test_index], predicted=predicted_y)
                correctness = (confusion_matrix.trace() / float(predicted_y.shape[0])) * 100
                learning_result.append(correctness)
                training_stat = get_performance_array(num_classes=feature.num_enabled_activities,
                                                      confusion_matrix=confusion_matrix)
                # Draw out the event graph
                # event_bar_plot(feature.time[test_index], feature.y[test_index], feature.num_enabled_activities,
                #                classified=predicted_y, max_days=10,
                #                ignore_activity=feature.activity_list['Other_Activity']['index'],
                #                legend=[feature.get_activity_by_index(i) for i in range(feature.num_enabled_activities)])
                overall_list_row += 1
                overall_sheet.write(overall_list_row, 0, datafile)
                overall_sheet.write(overall_list_row, 1, hidden_layer.__str__())
                overall_sheet.write(overall_list_row, 2, '%.5f' % correctness)
                overall_sheet.write(overall_list_row, 3, ('week %3d' % week_id))
                dataset_sheet = book.add_sheet(datafile + '_' +
                                               '_'.join([('%d' % hidden_layer[layer])
                                                         for layer in range(len(hidden_layer))]) +
                                               ('_week%03d' % week_id)
                                               )
                for c in range(0, len(dataset_list_title)):
                    dataset_sheet.write(0, c, str(dataset_list_title[c]))
                num_performance = len(performance_index)
                for r in range(feature.num_enabled_activities):
                    activity_label = feature.get_activity_by_index(r)
                    dataset_sheet.write(r+1, 0, activity_label)
                    for c in range(num_performance):
                        dataset_sheet.write(r+1, c+1, '%.5f' % training_stat[r][c])
                book.save('casas_sda_byweek.xls')
            learning_curve_fname = 'sda_learning_' + datafile + '_' + \
                                   '_'.join([('%d' % hidden_layer[layer]) for layer in range(len(hidden_layer))]) + \
                                   '.pkl'
            save_casas_learning_curve(learning_curve_fname,
                                      '%s sda %d of %d nodes' % (datafile, len(hidden_layer), hidden_layer[0]),
                                      learning_result, 'week')
    plot_casas_learning_curve(['sda_learning_b1_200_200_200.pkl'])
