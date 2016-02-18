import numpy as np
import time
import matplotlib.pyplot as plt
from actlearn.clustering.kmeans import kmeans_cluster


if __name__ == '__main__':
    # Create Sample Data in 2 dimensional array with 4 center location (1,1) (-1, 1) (1, -1), (-1, -1)
    sample = np.asarray(np.random.normal(0.0, 0.4, (500, 2)))
    dataset = np.concatenate((sample + [1, 1], sample + [-1, 1], sample + [1, -1], sample + [-1, -1]), axis=0)
    plt.figure(0)
    plt.scatter(dataset[:, 0], dataset[:, 1], color='g', edgecolor='k')
    cluster = kmeans_cluster(dataset, np.arange(2000), 4)
    color_array = ['c', 'r', 'b', 'k']
    for i in range(dataset.shape[0]):
        plt.scatter(dataset[i, 0], dataset[i, 1], color=color_array[cluster[i]], edgecolor='k')
    plt.show()
