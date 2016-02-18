cimport cython
import numpy as np
cimport numpy as np
from actlearn.log.logger import actlearn_logger

sort_logger = actlearn_logger.get_logger('utils.merge_sort')

cpdef index_sort(np.ndarray[double, ndim=2, mode="c"] data,
                np.ndarray[long, ndim=1, mode="c"] indices, int attribute):
    """
    Merge sort data array (data) with indices in indices array based on attribute column and
    return the sorted indices
    :type data: numpy.ndarray
    :param data: 2D array containing data
    :param indices:
    :param attribute:
    :return:
    """
    cdef double[:,:] data_ptr
    cdef long* indices_ptr
    cdef long* sorted_ptr
    cdef np.ndarray sorted_indices
    num_data = indices.shape[0]
    sorted_indices = np.zeros((num_data,), dtype=np.int)
    # Get Pointers of numpy.array
    data_ptr = data
    indices_ptr = <long *> indices.data
    sorted_ptr = <long *> sorted_indices.data
    # Print Some Data To make sure that it works
    index_msort(data_ptr, sorted_ptr, indices_ptr, num_data, attribute)
    return sorted_indices

cdef index_msort(double[:,:] data, long *tmp_array, long *train_array,
                 long num_train, long attribute):
    """
    Merge sort data array (self.x) based on attribute column and put the index in train_array
    without affecting data in self.x
    :param tmp_array:
    :param train_array:
    :param num_train:
    :param attribute:
    :return:
    """
    cdef unsigned long num_left = 0
    cdef unsigned long num_right = 0
    cdef unsigned long index_left = 0
    cdef unsigned long index_right = 0
    cdef unsigned long index = 0
    sort_logger.debug([train_array[i] for i in range(num_train)])
    # If there is only one item - no need to sort
    if num_train == 1:
        return
    # Sort two sub-array recursively
    num_left = num_train/2
    num_right = num_train - num_left
    # sort left
    index_msort(data, tmp_array, train_array, num_left, attribute)
    index_msort(data, tmp_array, &train_array[num_left], num_right, attribute)
    # Merge Left and Right
    index = 0
    index_left = 0
    index_right = num_left
    sort_logger.debug("merge: num_train %d" % num_train)
    sort_logger.debug([train_array[i] for i in range(num_train)])
    for index in range(num_train):
        # Compare Left and right
        if index_left == num_left:
            tmp_array[index] = train_array[index_right]
            index_right += 1
        elif index_right == num_train:
            # Left one is smaller, take left one
            tmp_array[index] = train_array[index_left]
            index_left += 1
        elif data[train_array[index_left]][attribute] > data[train_array[index_right]][attribute]:
            # Right one is smaller, take right one
            tmp_array[index] = train_array[index_right]
            index_right += 1
        else:
            # Left one is smaller, take left one
            tmp_array[index] = train_array[index_left]
            index_left += 1
    for index in range(num_train):
        train_array[index] = tmp_array[index]
