import time
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.collections as collections
from matplotlib.patches import Rectangle


default_color_list = ['yellow', 'blue', 'green',
                      'peru', 'blueviolet', 'darkorange', 'aqua',
                      'magenta', 'seagreen', 'lavender', 'khaki',
                      'lightpink', 'lime', 'gray', 'darkgoldenrod',
                      'dogerblue', 'deeppink']

def event_bar_plot(time_array, label, num_classes, classified=None, ignore_activity=-1):
    """
    :type time_array: np.array
    :param time_array: array of time in float format
    :type label: np.array
    :param label: array of class label (int)
    :param classified:
    :return:
    """
    # Get number of events
    num_events = time_array.shape[0]
    # Bar List
    bar_list = []
    # Date Tickers
    y = []
    y_label = []
    # Create Color List
    color_list = default_color_list
    if ignore_activity > -1 and ignore_activity < num_classes:
        color_list.insert(ignore_activity, 'white')
    # Count the days
    num_days = 0
    cur_datetime = datetime.fromtimestamp(time_array[0])
    cur_date = cur_datetime.date()
    label_start_x = (time_array[0] - time.mktime(cur_date.timetuple())) / float(24*60*60) * 2400
    prev_label = label[0]
    label_height = 30
    if classified is not None:
        label_height = 10
        prev_classified = classified[0]
        classified_start_x = label_start_x
        prev_error = False
        error_start_x = classified_start_x
        y.append(25)
        y_label.append('%2d.%2d orig' % (cur_date.month, cur_date.day))
        y.append(15)
        y_label.append('%2d.%2d pred' % (cur_date.month, cur_date.day))
        y.append(5)
        y_label.append('%2d.%2d err' % (cur_date.month, cur_date.day))
    else:
        y.append(15)
        y_label.append('%2d.%2d' % (cur_date.month, cur_date.day))
    i = 0
    while i < num_events and num_days < 31:
        # Log events plot locations
        cur_datetime = datetime.fromtimestamp(time_array[i])
        cur_label = label[i]
        if classified is not None:
            cur_classified = classified[i]
            cur_error = (classified[i] != label[i])
        # Check whether datetime advanced or not
        if cur_datetime.date() != cur_date:
            # End last day bars and then start checking labels
            label_length = 2400 - label_start_x
            bar_list.append(((label_start_x, (30+label_height)*num_days + 30 - label_height),
                             label_length, label_height, color_list[prev_label]))
            # Check classified
            if classified is not None:
                classified_length = 2400 - classified_start_x
                bar_list.append(((classified_start_x, (30+label_height)*num_days + 10), classified_length, 10, color_list[prev_label]))
                if prev_error == 1:
                    error_length = 2400 - error_start_x
                    bar_list.append(((error_start_x, (30+label_height)*num_days), error_length, 10, 'red'))
                classified_start_x = 0
                error_start_x = 0
                y.append((30+label_height)*(num_days+1)+25)
                y_label.append('%2d.%2d orig' % (cur_datetime.month, cur_datetime.day))
                y.append((30+label_height)*(num_days+1)+15)
                y_label.append('%2d.%2d pred' % (cur_datetime.month, cur_datetime.day))
                y.append((30+label_height)*(num_days+1)+5)
                y_label.append('%2d.%2d err' % (cur_datetime.month, cur_datetime.day))
            else:
                y.append((30+label_height)*(num_days+1)+15)
                y_label.append('%2d.%2d' % (cur_datetime.month, cur_datetime.day))
            label_start_x = 0
            cur_date = cur_datetime.date()
            num_days += 1
        # If label changed add rectangle position to bar_list
        if cur_label != prev_label:
            new_start_x = (time_array[i] - time.mktime(cur_datetime.date().timetuple())) / float(24*60*60) * 2400
            length = new_start_x - label_start_x
            bar_list.append(((label_start_x, (30+label_height)*num_days + 30 - label_height),
                             length, label_height, color_list[prev_label]))
            label_start_x = new_start_x
            prev_label = cur_label
        if classified is not None:
            if cur_classified != prev_classified:
                new_start_x = (time_array[i] - time.mktime(cur_datetime.date().timetuple())) / float(24*60*60) * 2400
                length = new_start_x - classified_start_x
                bar_list.append(((classified_start_x, (30+label_height)*num_days + 10),
                                 length, 10, color_list[prev_classified]))
                classified_start_x = new_start_x
                prev_classified = cur_classified
            if cur_error != prev_error:
                new_start_x = (time_array[i] - time.mktime(cur_datetime.date().timetuple())) / float(24*60*60) * 2400
                if prev_error:
                    length = new_start_x - error_start_x
                    bar_list.append(((error_start_x, (30+label_height)*num_days),
                                     length, 10, 'red'))
                error_start_x = new_start_x
                prev_error = cur_error
        i += 1
    # Assume time normalized
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlim([0, 2400])
    ax.set_ylim([0, num_days * (30 + label_height)])
    x = range(0, 2400, 100)
    x_label = ['%02d:00' % i for i in range(24)]
    ax.set_xticks(x)
    ax.set_xticklabels(x_label, rotation='vertical')
    # y = range(0, 1000, 200)
    # y_label = ['class1', 'class2', 'class3', 'class4', 'class5']
    ax.set_yticks(y)
    ax.set_yticklabels(y_label)
    for bar in bar_list:
        ax.add_artist(Rectangle(bar[0], bar[1], bar[2], color=bar[3]))
    plt.show()
