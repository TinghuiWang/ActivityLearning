import os
import cPickle as pickle
import matplotlib.pyplot as plt
from actlearn.log.logger import actlearn_logger
from actlearn.utils.AlResult import AlResult
from actlearn.utils.classifier_performance import *


def plot_result_time_series(filename_array, fig_fname='', fig_format='pdf',
                            performance_name='overall_performance', performance_key='exact matching ratio'):
    """
    Show the learning result on a time-series plot.
    :param filename_array:
    :param fig_fname:
    :param fig_format:
    :param performance_name: overall_performance or per_class_performance
    :param performance_key: name listed in overall_performance or per_class_performance
    :return:
    """
    logger = actlearn_logger.get_logger('casas_learning_curve')
    # Load data into array of dictionary
    load_error = False
    learning_data = []
    for fname in filename_array:
        if os.path.exists(fname):
            cur_data = AlResult()
            cur_data.load_result(fname=fname)
            learning_data.append(cur_data)
        else:
            logger.error('Cannot find file %s' % fname)
            load_error = True
    if load_error:
        return
    # Check the base of all learning curve files
    # 1. Same length?
    # 2. By week or by day?
    result_mode = learning_data[0].get_mode()
    curve_length = learning_data[0].get_num_results()
    curve_x_label = learning_data[0].get_record_keys()
    for result_data in learning_data:
        if result_data.get_mode() != result_mode:
            load_error = True
        cur_length = result_data.get_num_results()
        if cur_length != curve_length:
            logger.warning('In consistence length of learning curve: %s has %d items' %
                           (result_data.get_name(), cur_length))
            if cur_length < curve_length:
                curve_length = cur_length
                curve_x_label = result_data.get_record_keys()
    if load_error:
        logger.error('Error in step size of learning curve')
        for curve in learning_data:
            logger.error('%20s: by %s' % (curve.get_name(), curve.get_mode()))
        return
    # Check if performance key is legal
    performance_key_id = 0
    if performance_name == 'per_class_performance':
        if performance_key not in per_class_performance_index:
            logger.error('key %s not found in per_class_performance.\nAvailable Keys are: %s' %
                         (performance_key, str(per_class_performance_index)))
            return
        else:
            performance_key_id = per_class_performance_index.index(performance_key)
    elif performance_name == 'overall_performance':
        if performance_key not in overall_performance_index:
            logger.error('key %s not found in overall_performance.\nAvailable Keys are: %s' %
                         (performance_key, str(overall_performance_index)))
            return
        else:
            performance_key_id = overall_performance_index.index(performance_key)
    else:
        logger.error('performance matrix named %s is not available' % performance_name)
        return
    # Synthesize the selected performance value into an array that can be plotted
    curve_value = []
    curve_name = []
    for result_data in learning_data:
        curve_name.append(result_data.get_name())
        cur_curve = []
        for record_key in result_data.get_record_keys():
            cur_curve.append(result_data.get_result_by_key(record_key)[performance_name][performance_key_id])
        curve_value.append(cur_curve)
    # Plot it using plt
    x = range(curve_length)
    x_label = curve_x_label
    y = np.arange(0, 1.1, 0.1)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for i in range(len(curve_name)):
        plt.plot(x, curve_value[i], label=curve_name[i])
    ax.set_xticks(x)
    ax.set_xticklabels(x_label, rotation='vertical')
    ax.set_yticks(y)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels)
    if fig_fname != '':
        fig.savefig(fig_fname, transparent=True, format=fig_format, bbox_inches='tight')
    else:
        plt.show()
    return

