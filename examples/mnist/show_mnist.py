from actlearn.data.mnist import *
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import wx


def get_path(wildcard):
    app = wx.App(None)
    style = wx.FD_OPEN | wx.FD_FILE_MUST_EXIST
    dialog = wx.FileDialog(None, 'Open', wildcard=wildcard, style=style)
    if dialog.ShowModal() == wx.ID_OK:
        path = dialog.GetPath()
    else:
        path = None
    dialog.Destroy()
    return path

if __name__ == '__main__':
    file_path = get_path('*.pkl')
    print('Loading MNIST data from %s' % file_path)
    train_set, valid_set, test_set = mnist_load_data(file_path)
    (train_set_x, train_set_y) = train_set
    (valid_set_x, valid_set_y) = valid_set
    (test_set_x, test_set_y) = test_set

    train_set_len = train_set_x.shape[0]
    valid_set_len = valid_set_x.shape[0]
    test_set_len = test_set_x.shape[0]
    total_index = train_set_len + valid_set_len + test_set_len
    print('%d data loaded' % total_index)

    user_input = 0
    while user_input != -1:
        data_found = True
        user_input = int(input('Input an index (max: %d; -1 for quit): ' % total_index))
        if user_input < train_set_len:
            img = train_set_x[user_input].reshape(28, 28)
            label = train_set_y[user_input]
        elif user_input < train_set_len + valid_set_len:
            img = valid_set_x[user_input - train_set_len].reshape(28, 28)
            label = valid_set_y[user_input - train_set_len]
        elif user_input < total_index:
            img = test_set_x[user_input - train_set_len - valid_set_len].reshape(28, 28)
            label = test_set_y[user_input - train_set_len - valid_set_len]
        else:
            data_found = False
            label = -1
            img = None
            print('The index should be smaller than %d. Please enter a smaller one.' % total_index)
        if data_found:
            print('class: %d' % label)
            plt.figure(1)
            plt.imshow(img, cmap=cm.gray)
            plt.show()
