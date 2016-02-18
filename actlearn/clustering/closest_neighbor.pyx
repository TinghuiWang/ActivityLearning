cimport cython
import numpy as np
cimport numpy as np
from actlearn.log.logger import actlearn_logger


cpdef closest_neighbor(np.ndarray[double, ndim=2, mode="c"] data,
                       np.ndarray[long, ndim=1, mode="c"] indices, int k):
    """
    :param data: Data Array (Double)
    :param indices: indices of data that needs to be organized into clusters
    :param k: number of clusters
    :return:
    """
    cdef double[:,:] data_ptr
    cdef long[:] indices_ptr
    cdef long num_data
    cdef np.ndarray cluster
    cdef long[:] cluster_view
    cdef object vertices_array = []

    num_data = indices.shape[0]
    # Initialize data structure that holds cluster indices
    cluster = np.zeros((num_data,), dtype=np.int)
    cluster_view = cluster
    # Get memor view of numpy.array
    data_ptr = data
    indices_ptr = indices
    temp = indices.copy()
    # Pick the first data point where it is farthest from the center
    # Calculate the center first
    avg_vector = np.mean(data[indices], axis=0)
    print(avg_vector)
    # pick first data point that is farthest to the middle of the dataset
    distance = np.sum((data - avg_vector) ** 2, axis=1)
    max_distance_id = distance[indices].argmax()
    vertices_array.append(indices[max_distance_id])
    # Temporary indices except for the one in the vertices array
    temp = np.delete(temp, max_distance_id)
    # pick the next k-1 vertices that has the maximum distance between all nodes
    for i in range(k-1):
        distance[:] = 0
        for index in vertices_array:
            distance += np.sum((data - data[index][:]) ** 2, axis=1)
        max_distance_id = distance[temp].argmax()
        vertices_array.append(temp[max_distance_id])
        temp = np.delete(temp, max_distance_id)
    # For each remaining items find the distance to four vertices and decide which cluster it belongs to
    cur_distance = np.zeros((k,), dtype=np.double)
    for i in range(num_data):
        cur_distance[:] = 0.
        for vertex_index, vertex in enumerate(vertices_array):
            cur_distance[vertex_index] = np.sum((data[indices_ptr[i], :] - data[vertex, :]) ** 2)
        cluster_id = cur_distance.argmin()
        cluster[i] = cluster_id
    return cluster
