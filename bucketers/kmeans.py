"""
Uniformly chunks a matrix into 'num_buckets' with no reordering. If no 
'num_buckets' is specified this will return the entire dimension in one bucket.
"""
import numpy as np
import time
from .bucketer import Bucketer
from sklearn.cluster import KMeans
from tqdm import tqdm


class KmeansRowBucketer(Bucketer):

    def __init__(self, num_buckets):
        self.num_buckets = num_buckets

    def name(self):
        return "uniform"

    def bucket(self, X):
        assert (self.num_buckets <= X.shape[0])

        # Run kmeans to determine the buckets.
        t0 = time.time()
        kmeans = KMeans(n_clusters=self.num_buckets).fit(X)
        print("KMeans Bucket Rows Time: " + str(time.time() - t0))

        buckets = []
        bucket_index = np.zeros(self.num_buckets, dtype=np.int)
        bincount = np.bincount(kmeans.labels_)
        
        sum_prev_indexes = np.zeros(bincount.shape, dtype=np.int)
        final_indexes = np.zeros(X.shape[0], dtype=np.int)
        running_sum = 0
        for i in range(len(bincount)):
            bc = bincount[i]
            buckets.append(np.zeros([bc, X.shape[1]]))
            sum_prev_indexes[i] = running_sum
            running_sum += bc
        for i in range(len(kmeans.labels_)):
            label = kmeans.labels_[i]
            bucket = buckets[label]
            bucket[bucket_index[label], :] = X[i, :]
            final_indexes[bucket_index[label] + sum_prev_indexes[label]] = i
            bucket_index[label] += 1

        return buckets, final_indexes
