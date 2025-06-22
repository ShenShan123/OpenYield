import numpy as np
from Distribution.utils import normalize, cos_distance
import time

class cone_cluster():
    def __init__(self, cluster_num, dim, origin_centroids=None):
        if isinstance(origin_centroids, type(None)):
            origin_centroids = np.random.normal(loc=0, scale=1, size=[cluster_num, dim])
            origin_centroids = normalize(origin_centroids) # return unit vector
        self.k = cluster_num
        self.dim = dim
        self.V = origin_centroids

    def cluster(self, x):
        start_t = time.time()
        stop_flag = False
        last_labels = -1 * np.ones([x.shape[0]])
        while stop_flag == False:
            cos_matrix = cos_distance(x, self.V)
            now_labels = np.argmin(cos_matrix, axis=-1)
            for i in range(self.k):
                self.V[i,:] = x[now_labels==i,:].mean(axis=0)
            if (last_labels == now_labels).all():
                stop_flag = True
            last_labels = now_labels

            now_t = time.time()
            if now_t - start_t > 3: # in case of infinite iterations
                print("infinite iterations")
                break

        return now_labels


if __name__ == "__main__":
    t = cone_cluster(cluster_num=5, dim=5)
    x = np.random.normal(0,1,[100,5])
    now_labels = t.cluster(x)
    print(now_labels)