cimport cython
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
from libc.stdio cimport stdout
from cpython.string cimport PyString_AsString
import numpy as np
cimport numpy as np
from actlearn.log.logger import actlearn_logger

from actlearn.models.model import Model

cdef struct DecisionTreeNode:
    DecisionTreeNode* parent
    DecisionTreeNode** children
    int numChildren
    int attribute
    unsigned long numInstances
    unsigned long *instances
    float threshold
    int classId
    int numRight
    float entropy
    float *adist


cdef struct SplitInfo:
    int valid
    float threshold
    float infoGain
    float gainRatio

cdef class DecisionTree:
    """
    DecisionTree class implements model interface for decision tree
    structure.

    The class is callable in python codes, but cdef parameters are
    not accessible.

    Parameters:
    root: Pointer to DecisionTreeNode Root
    activity_num: Number of Activities (class label from 0 to activity_num - 1)
    feature_num: Number of Features
    """
    cdef DecisionTreeNode* root
    cdef int activity_num
    cdef int feature_num
    cdef np.ndarray x
    cdef np.ndarray y
    cdef np.ndarray train_array
    cdef unsigned long training_num
    cdef int minimum_object_of_leaf

    cdef float *probability_left
    cdef float *probability_right

    cdef float *possible_threshold

    def __cinit__(self, feature_num, activity_num):
        """
        Intialize Data Structure. Callable in Python.
        :param feature_num: Enabled Total Feature Count
        :param activity_num: Enabled Activity Count
        :return:
        """
        self.root = NULL
        self.activity_num = activity_num
        self.feature_num = feature_num
        self.train_array = None
        self.training_num = 0
        self.minimum_object_of_leaf = 2
        self.logger = actlearn_logger.get_logger('DecisionTree')

    def build(self, np.ndarray x, np.ndarray y):
        """
        :param x:
        :param y:
        :return:
        """
        self.x = x
        self.y = y
        self.training_num = self.x.shape[1]
        self.train_array = np.arange(self.training_num, dtype=np.dtype('uint32'))
        self.root = self.build_tree_node(self.train_array, 0, self.training_num, NULL)
        self.probability_left = <float *> PyMem_Malloc(self.activity_num*sizeof(float))
        self.probability_right = <float *> PyMem_Malloc(self.activity_num*sizeof(float))
        self.possible_threshold = <float *> PyMem_Malloc(self.training_num*sizeof(float))

    cdef DecisionTreeNode * build_tree_node(self, np.ndarray train_array, unsigned long start_pos,
                         unsigned long num_train, DecisionTreeNode * parent):
        """
        Recursive Method for building tree node
        
        :param num_train: number of training samples for current node
        :param parent: pointer to parent node. NULL if it is the root node
        :return:
        """
        cdef DecisionTreeNode *node
        cdef int major_class_id
        cdef int no_split
        cdef unsigned long major_class_count
        # Allocate space for current node
        node = self.allocate_tree_node()
        # Set Parents
        node.parent = parent
        # Records the number of training instances within the tree
        node.numInstances = num_train
        if num_train == 0:
            # No more training examples left, make it leaf
            node.classId = node.parent.classId
        else:
            # Populate array with a list of training example IDs
            node.instances = <unsigned long *> PyMem_Malloc(num_train * sizeof(unsigned long))
            for i in range(num_train):
                node.instances[i] = train_array[i]
            # Find Majority Class, whether to further split current node and the count of the majority class
            (major_class_id, no_split, major_class_count) = self.find_major_class(train_array, start_pos, num_train)
            node.classId = major_class_id
            node.numRight = major_class_count
            # If all examples in training array belong to same class or examples too small
            # make current node a leaf node (return here directly)
            # Otherwise, find the attribute and best split and recursively build the sub-tree
            if major_class_count != num_train and no_split != 0:
                self.select_attributes(train_array, start_pos, major_class_count, node)
                # Recursively construct sub-tree




    cdef select_attributes(self, np.ndarray train_array, unsigned long start_pos,
                           unsigned long num_train, DecisionTreeNode* dt_node):
        """
        Select the attribute and threshold of the split point for current
        node according to the best information gain ratio
        :param train_array: array of indexes to data
        :param num_train: number of training examples
        :return:
        """
        cdef SplitInfo* split_info_array
        cdef unsigned long attribute_index
        cdef float avg_info_gain
        cdef unsigned long valid_split_count
        cdef unsigned long best_attribute_index
        cdef float best_gain_ratio
        cdef float best_threshold
        # allocate memory for storing information in order to find the best split
        split_info_array = <SplitInfo *> PyMem_Malloc(self.feature_num*sizeof(SplitInfo))
        # Go through every attribute and calculate their information gain with best numeric split
        valid_split_count = 0
        for attribute_index in range(self.feature_num):
            self.find_best_numeric_split(train_array, start_pos, num_train, dt_node, attribute_index, split_info_array)
            if split_info_array[attribute_index].valid != 0:
                avg_info_gain += split_info_array[attribute_index].infoGain
                valid_split_count += 1
        # Search through all features to find the best feature and split point with the best information gain
        best_gain_ratio = 0.
        if valid_split_count > 0:
            avg_info_gain = avg_info_gain/valid_split_count
            for attribute_index in range(self.feature_num):
                if split_info_array[attribute_index].valid != 0 and \
                    split_info_array[attribute_index].infoGain >= avg_info_gain and \
                    split_info_array[attribute_index].gainRatio > best_gain_ratio:
                    best_attribute_index = attribute_index
                    best_threshold = split_info_array[attribute_index].threshold
                    best_gain_ratio = split_info_array[attribute_index].gainRatio
        # Free temporary memory
        PyMem_Free(split_info_array)
        # Update Info for current tree node
        if best_gain_ratio > 0.:
            dt_node.attribute = best_attribute_index
            """Need to Update the threshold calculation function C45SplitPoint"""
            dt_node.threshold = best_threshold
            if self.debug > 1:
                print("select_attributes: best attribute %d, threshold %f" %
                      (best_attribute_index, best_threshold))

    cdef find_best_numeric_split(self, np.ndarray train_array, unsigned long start_pos,
                                 unsigned long num_train, DecisionTreeNode* dt_node,
                                 unsigned long attribute_index, SplitInfo* split_info_array):
        """
        Find the best numeric split for a specific attribute
        and calculate the corresponding information gain
        :param train_array:
        :param num_train:
        :param dt_node:
        :param attribute_index:
        :param split_info_array:
        :return:
        """
        cdef float last_threshold
        cdef float current_threshold
        cdef unsigned long border
        cdef unsigned long left_count
        cdef unsigned long right_count
        cdef float entropy_left
        cdef float entropy_right

        split_info_array[attribute_index].valid = 0
        # Sort train_array according to some attribute
        current_index_array = train_array[np.arange(num_train)]
        sorted_index_array = self.x[current_index_array][:,attribute_index].argsort(axis=0, kind='mergesort')
        np.place(train_array, np.arange(self.training_num)<num_train, current_index_array[sorted_index_array])

        # Find the number of threshold and add the number to the threshold array
        minimum_split = max(min(0.1 * (num_train / self.activity_num), 25), 2)
        threshold_count = 1
        last_threshold = self.x[train_array[start_pos]][attribute_index]
        self.possible_threshold[0] = last_threshold
        for i in range(start_pos+1, start_pos+num_train):
            current_threshold = self.x[train_array[i]][attribute_index]
            if last_threshold < current_threshold and \
                i >= minimum_split and (num_train - i) >= minimum_split:
                self.possible_threshold[threshold_count] = (current_threshold + last_threshold)/2
                if self.possible_threshold[threshold_count] == last_threshold:
                    self.possible_threshold[threshold_count] = last_threshold
                threshold_count += 1
            last_threshold = current_threshold

        # Go through each threshold, calculate the information gain and return the best threshold
        border = 0
        left_count = 0
        right_count = num_train
        for i in range(self.activity_num):
            self.probability_left[i] = 0.
            self.probability_right[i] = 0.
        for i in range(start_pos, start_pos+num_train):
            self.probability_right[self.y[i]] += 1
        for i in range(threshold_count):
            while border < num_train and \
                self.x[train_array[border]][attribute_index] <= self.possible_threshold[i]:
                self.probability_left[self.y[train_array[border]]] += 1
                self.probability_right[self.y[train_array[border]]] -= 1
                left_count += 1
                right_count -= 1
                entropy_left = 0.
                entropy_right = 0.
                #for j in range(self.activity_num):
                #    entropy_left -= self.probability_left[j]/left_count * log2f(self.probability_left[j]/left_count)
        pass

    cdef entropy(self, ):
        pass

    cdef find_major_class(self, np.ndarray train_array, unsigned long start_pos,
                          unsigned long num_train):
        """
        Find the majority class in the training array and return with
        a tuple of class id, whether it smaller than the number of threshold,
        and the frequency of the majority class
        :param train_array: array of training examples
        :param num_train: number of training samples in the array
        :return: tuple (int classId, bool is_smaller, int frequency)
        """
        cdef unsigned long *activity_frequency
        cdef unsigned long max_frequency = 0
        cdef int classId = -1
        cdef int is_smaller = 0

        activity_frequency = <unsigned long *> PyMem_Malloc(self.activity_num * sizeof(unsigned long))
        for i in range(self.activity_num):
            activity_frequency[i] = 0

        for i in range(start_pos, start_pos+num_train):
            activity_frequency[self.y[train_array[i]]] += 1

        for i in range(self.activity_num):
            if activity_frequency[i] > max_frequency:
                classId = i
                max_frequency = activity_frequency[i]

        if max_frequency > self.minimum_object_of_leaf:
            is_smaller = 1
        else:
            is_smaller = 0

        PyMem_Free(activity_frequency)

        return classId, is_smaller, max_frequency

    cdef DecisionTreeNode * allocate_tree_node(self):
        cdef DecisionTreeNode *node
        node = <DecisionTreeNode *> PyMem_Malloc(sizeof(DecisionTreeNode))
        node.parent = NULL
        node.children = NULL
        node.numChildren = 0
        node.attribute = -1
        node.numInstances = 0
        node.instances = NULL
        node.threshold = 0
        node.classId = -1
        node.numRight = 0
        node.entropy = 0
        node.adist = NULL

    def classify(self, y):
        pass
