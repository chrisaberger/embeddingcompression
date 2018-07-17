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
    def __init__(self, num_buckets, max_num_buckets):
        Bucketer.__init__(self, num_buckets, max_num_buckets)

    def name(self):
        return "kmeans" + str(self.num_buckets)

    def extra_bytes_needed(self):
        # Just need to store the offset for each bucket.
        return self.num_buckets * 4

    def bucket(self, X):
        assert (self.num_buckets <= X.shape[0])

        # Run kmeans to determine the buckets.
        t0 = time.time()
        print("Running Kmeans to bucket rows...")
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


class KmeansColBucketer(Bucketer):

    def __init__(self, num_buckets, max_num_buckets):
        Bucketer.__init__(self, num_buckets, max_num_buckets)

    def name(self):
        return "kmeans" + str(self.num_buckets)

    def extra_bytes_needed(self):
        # In each bucket you need a mapping back to the original column order.
        return self.num_buckets * self.num_cols * 4 

    def bucket(self, row_buckets, X):
        self.num_cols = X.shape[1]
        final_indexes = []
        final_buckets = []
        for bucket in row_buckets:
            transposed_bucket = np.transpose(bucket)
            rowBucketer = KmeansRowBucketer(self.num_buckets, self.max_num_buckets)
            col_buckets, col_indexes = rowBucketer.bucket(transposed_bucket)
            for i in range(len(col_buckets)):
                col_buckets[i] = np.transpose(col_buckets[i])
            final_indexes.append(col_indexes)
            final_buckets.append(col_buckets)
        return final_buckets, final_indexes




