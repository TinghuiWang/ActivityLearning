import os
import time
import theano
import theano.tensor as T
import cPickle as pickle
import numpy as np
import xlwt
import pydot
from actlearn.data.casas import load_casas_from_file, get_boundary
from actlearn.models.StackedDenoisingAutoencoder import StackedDenoisingAutoencoder
from actlearn.utils.confusion_matrix import get_confusion_matrix
from actlearn.utils.classifier_performance import *
from actlearn.data.AlFeature import AlFeature
from actlearn.utils.event_bar_plot import event_bar_plot
from actlearn.utils.AlResult import AlResult

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
        week_array = get_boundary(feature, period='week')
        # Number of perceptrons in hidden layer
        hidden_layer_list = [[200, 200, 200]]
        for hidden_layer in hidden_layer_list:
            input_x = T.matrix('x')
            num_classes = feature.num_enabled_activities
            numpy_rng = np.random.RandomState(int(time.clock()))
            # Set Training Rate and Corruption Levels
            pre_training_learning_rate = []
            for i in range(2*len(hidden_layer)):
                pre_training_learning_rate.append(0.5)
            fine_tuning_training_rate = []
            for i in range(2*len(hidden_layer) + 2):
                fine_tuning_training_rate.append(0.1)
            corruption_levels = []
            for i in range(len(hidden_layer)):
                corruption_levels.append(0)
            pre_training_epochs = int(len(hidden_layer) * hidden_layer[0] / 30)
            fine_tuning_epochs = int(len(hidden_layer) * hidden_layer[0] / 10)
            # Create Model
            model = StackedDenoisingAutoencoder(numpy_rng=numpy_rng, input=input_x,
                                                n_ins=feature.x.shape[1], n_outs=num_classes,
                                                hidden_layers_sizes=hidden_layer,
                                                corruption_levels=corruption_levels)
            for week_id in range(1, 10): # (len(week_array) - 1):
                next_week_result_fname = 'sda_%s_week_%d_200_200_200_loss_nxt_week_perform.pkl' % (datafile, week_id)
                this_week_result_fname = 'sda_%s_week_%d_200_200_200_loss_this_week_perform.pkl' % (datafile, week_id)
                prev_week_result_fname = 'sda_%s_week_%d_200_200_200_loss_prev_week_perform.pkl' % (datafile, week_id)
                next_week_result_name = 'next week prediction'
                this_week_result_name = 'this week'
                prev_week_result_name = 'previous week'
                next_week_result = AlResult(result_name=next_week_result_name, data_fname=datafile, mode='vs_loss')
                this_week_result = AlResult(result_name=this_week_result_name, data_fname=datafile, mode='vs_loss')
                prev_week_result = AlResult(result_name=prev_week_result_name, data_fname=datafile, mode='vs_loss')
                if os.path.exists(prev_week_result_fname):
                    prev_week_result.load_from_file(prev_week_result_fname)
                if os.path.exists(this_week_result_fname):
                    this_week_result.load_from_file(this_week_result_fname)
                if os.path.exists(next_week_result_fname):
                    next_week_result.load_from_file(next_week_result_fname)
                # After this point, all result structure loaded or created
                # Pick the one with most records as the start point
                prev_week_num_records = prev_week_result.get_num_records()
                this_week_num_records = this_week_result.get_num_records()
                next_week_num_records = next_week_result.get_num_records()
                prev_week_record_keys = prev_week_result.get_record_keys()
                this_week_record_keys = this_week_result.get_record_keys()
                next_week_record_keys = next_week_result.get_record_keys()
                max_records_len = max(prev_week_num_records, this_week_num_records, next_week_num_records)
                if prev_week_num_records == max_records_len:
                    reference_record_keys = prev_week_record_keys
                    reference_result = prev_week_result
                elif this_week_num_records == max_records_len:
                    reference_record_keys = this_week_record_keys
                    reference_result = this_week_result
                else:
                    reference_record_keys = next_week_record_keys
                    reference_result = next_week_result
                # Setup up the index range of this week, last week or next week.
                if week_id == 1:
                    prev_week_index = range(0, week_array[week_id - 1])
                else:
                    prev_week_index = range(week_array[week_id - 2], week_array[week_id - 1])
                this_week_index = range(week_array[week_id - 1], week_array[week_id])
                next_week_index = range(week_array[week_id], week_array[week_id + 1])
                # if all three agree on the first data records, then no need to do pre_training
                if max_records_len == 0:
                    # Performance Evaluation
                    model.do_pretraining(data=x_tensor[this_week_index],
                                         num_data=len(this_week_index), batch_size=10,
                                         learning_rate_array=pre_training_learning_rate,
                                         num_epochs=pre_training_epochs)
                for i in range(fine_tuning_epochs):
                    if i < max_records_len:
                        record_key = reference_record_keys[i]
                        model.load_from_dict(reference_result.get_record_by_key(record_key)['model'])
                    else:
                        result = model.do_fine_tuning(data=x_tensor[this_week_index], label=y_tensor[this_week_index],
                                                      num_data=len(this_week_index), batch_size=10,
                                                      learning_rate_array=fine_tuning_training_rate,
                                                      num_epochs=1)
                        record_key = result[0]
                    if record_key not in prev_week_record_keys:
                        (predicted_y,) = model.classify(x_tensor[prev_week_index])
                        confusion_matrix = get_confusion_matrix(num_classes=feature.num_enabled_activities,
                                                                label=feature.y[prev_week_index], predicted=predicted_y)
                        (overall_performance, per_class_performance) = \
                            get_performance_array(num_classes=feature.num_enabled_activities,
                                                  confusion_matrix=confusion_matrix)
                        prev_week_result.add_record(model.export_to_dict(), key=record_key,
                                                    confusion_matrix=confusion_matrix,
                                                    overall_performance=overall_performance,
                                                    per_class_performance=per_class_performance)
                        prev_week_result.save_to_file(prev_week_result_fname)
                    if record_key not in this_week_record_keys:
                        (predicted_y,) = model.classify(x_tensor[this_week_index])
                        confusion_matrix = get_confusion_matrix(num_classes=feature.num_enabled_activities,
                                                                label=feature.y[this_week_index], predicted=predicted_y)
                        (overall_performance, per_class_performance) = \
                            get_performance_array(num_classes=feature.num_enabled_activities,
                                                  confusion_matrix=confusion_matrix)
                        this_week_result.add_record(model.export_to_dict(), key=record_key,
                                                    confusion_matrix=confusion_matrix,
                                                    overall_performance=overall_performance,
                                                    per_class_performance=per_class_performance)
                        this_week_result.save_to_file(this_week_result_fname)
                    if record_key not in next_week_record_keys:
                        (predicted_y,) = model.classify(x_tensor[next_week_index])
                        confusion_matrix = get_confusion_matrix(num_classes=feature.num_enabled_activities,
                                                                label=feature.y[next_week_index], predicted=predicted_y)
                        (overall_performance, per_class_performance) = \
                            get_performance_array(num_classes=feature.num_enabled_activities,
                                                  confusion_matrix=confusion_matrix)
                        next_week_result.add_record(model.export_to_dict(), key=record_key,
                                                    confusion_matrix=confusion_matrix,
                                                    overall_performance=overall_performance,
                                                    per_class_performance=per_class_performance)
                        next_week_result.save_to_file(next_week_result_fname)

