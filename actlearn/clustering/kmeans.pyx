cimport cython
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
import numpy as np
cimport numpy as np
from actlearn.log.logger import actlearn_logger

cpdef kmeans_cluster(np.ndarray[double, ndim=2, mode="c"] data,
                     np.ndarray[long, ndim=1, mode="c"] indices, int k):
    """
    :param data: Data Array (Double)
    :param indices: indices of data that needs to be organized into clusters
    :param k: number of clusters
    :return:
    """
    cdef long** cluster_buffer
    cdef long* cluster_count
    cdef double[:,:] data_ptr
    cdef long[:] indices_ptr
    cdef np.ndarray distance_array
    cdef int cur_cluster_buffer
    cdef int num_cluster_changed
    cdef long num_data
    cdef long end_iteration
    cdef np.ndarray cluster
    cdef long[:] cluster_view

    num_data = indices.shape[0]
    # Set up memory for logging cluster id for each data points
    cluster_buffer = <long **> PyMem_Malloc(2 * sizeof(long *))
    cluster_buffer[0] = <long *> PyMem_Malloc(num_data * sizeof(long))
    cluster_buffer[1] = <long *> PyMem_Malloc(num_data * sizeof(long))
    for i in range(num_data):
        cluster_buffer[0][i] = -1
        cluster_buffer[1][i] = -1
    cluster_count = <long *> PyMem_Malloc(k * sizeof(long))
    cluster = np.zeros((num_data,), dtype=np.int)
    cluster_view = cluster
    indices_ptr = indices
    # Random choose 10 centers
    cluster_center = np.random.uniform(0, 1, size=(k, data.shape[1]))
    # Get memory view of data
    data_ptr = data
    # Now start iteration
    end_iteration = 0
    cur_cluster_buffer = 0
    while end_iteration == 0:
        # Assign numbers to each cluster with defined centers
        for i in range(num_data):
            distance_array = np.sum((cluster_center - data[indices_ptr[i]]) ** 2, axis=1)
            cluster_buffer[cur_cluster_buffer][i] = distance_array.argmin()
        # See if any data changed cluster - if so, keep iterating
        num_cluster_changed = 0
        for i in range(num_data):
            if cluster_buffer[0][i] != cluster_buffer[1][i]:
                num_cluster_changed += 1
        if num_cluster_changed == 0:
            # finish clustering
            end_iteration = 1
        else:
            # Recalculate the center
            for i in range(k):
                cluster_count[i] = 0
            cluster_center[:,:] = 0.
            for i in range(num_data):
                cluster_center[cluster_buffer[cur_cluster_buffer][i]][:] += data[indices_ptr[i]]
                cluster_count[cluster_buffer[cur_cluster_buffer][i]] += 1
            for i in range(k):
                if cluster_count[i] != 0:
                    cluster_center[i,:] = cluster_center[i,:] / cluster_count[i]
            cur_cluster_buffer = 1 - cur_cluster_buffer
    # Iteration End
    for i in range(num_data):
        cluster_view[i] = cluster_buffer[cur_cluster_buffer][i]
    # Free assigned nodes
    PyMem_Free(cluster_buffer[0])
    PyMem_Free(cluster_buffer[1])
    PyMem_Free(cluster_buffer)
    PyMem_Free(cluster_count)
    return cluster
