import os
import sys
import time
import theano
import theano.tensor as T
import cPickle as pickle
import numpy as np
import xlwt
import pydot
from actlearn.data.casas import load_casas_from_file
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
        for j in range(num_samples):
            if j % 3 == 0:
                test_index.append(j)
            else:
                train_index.append(j)
            num_fold = 3
        train_x = feature.x[train_index][:]
        train_y = feature.y[train_index]
        train_x_tensor = theano.shared(np.asarray(train_x, dtype=theano.config.floatX), borrow=True)
        train_y_tensor = T.cast(theano.shared(train_y, borrow=True), 'int32')
        test_x = feature.x[test_index][:]
        test_x_tensor = theano.shared(np.asarray(test_x, dtype=theano.config.floatX), borrow=True)
        # Number of perceptrons in hidden layer
        hidden_layer_list = [[200, 200, 200]]
        for hidden_layer in hidden_layer_list:
            input_x = T.matrix('x')
            num_classes = feature.num_enabled_activities
            numpy_rng = np.random.RandomState(int(time.clock()))
            model = StackedDenoisingAutoencoder(numpy_rng=numpy_rng, input=input_x,
                                                n_ins=train_x.shape[1], n_outs=num_classes,
                                                hidden_layers_sizes=hidden_layer,
                                                corruption_levels=[0, 0, 0])
            # Load MLP if exists
            sda_filename = datafile + '_' + \
                '_'.join([('%d' % hidden_layer[layer]) for layer in range(len(hidden_layer))]) + \
                '.pkl'
            if os.path.exists(sda_filename):
                # Load Model from File
                model.load(sda_filename)
            else:
                model.do_pretraining(data=train_x_tensor,
                                     num_data=train_x.shape[0], batch_size=10,
                                     learning_rate_array=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                                     num_epochs=15)
                model.do_fine_tuning(data=train_x_tensor, label=train_y_tensor,
                                     num_data=train_x.shape[0], batch_size=10,
                                     learning_rate_array=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                                     num_epochs=36)
                model.save(sda_filename)
            # model.do_fine_tuning(data=train_x_tensor, label=train_y_tensor,
            #                      num_data=train_x.shape[0], batch_size=10,
            #                      learning_rate_array=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
            #                      num_epochs=36)
            # Performance Evaluation
            result = model.classify(test_x_tensor)
            predicted_y = result[0]
            confusion_matrix = get_confusion_matrix(num_classes=feature.num_enabled_activities,
                                                    label=feature.y[test_index], predicted=predicted_y)
            correctness = (confusion_matrix.trace() / float(predicted_y.shape[0])) * 100
            training_stat = get_performance_array(num_classes=feature.num_enabled_activities,
                                                  confusion_matrix=confusion_matrix)
            # Draw out the event graph
            event_bar_plot(feature.time, feature.y, feature.num_enabled_activities, classified=predicted_y,
                           ignore_activity=feature.activity_list['Other_Activity']['index'])
            overall_list_row += 1
            overall_sheet.write(overall_list_row, 0, datafile)
            overall_sheet.write(overall_list_row, 1, hidden_layer.__str__())
            overall_sheet.write(overall_list_row, 2, '%.5f' % correctness)
            dataset_sheet = book.add_sheet(datafile + '_' +
                                           '_'.join([('%d' % hidden_layer[layer])
                                                     for layer in range(len(hidden_layer))]))
            for c in range(0, len(dataset_list_title)):
                dataset_sheet.write(0, c, str(dataset_list_title[c]))
            num_performance = len(performance_index)
            for r in range(feature.num_enabled_activities):
                activity_label = feature.get_activity_by_index(r)
                dataset_sheet.write(r+1, 0, activity_label)
                for c in range(num_performance):
                    dataset_sheet.write(r+1, c+1, '%.5f' % training_stat[r][c])
            book.save('casas_sda.xls')
            dot_string = model.export_to_graphviz([feature.get_feature_by_index(fid)
                                                   for fid in range(train_x.shape[1])])
            dot_filename = datafile + '_' + \
                '_'.join([('%d' % hidden_layer[layer]) for layer in range(len(hidden_layer))]) + \
                '.dot'
            dot_file = open(dot_filename, mode='w')
            dot_file.write(dot_string)
            dot_file.close()
