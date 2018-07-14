"""
Uniformly chunks a matrix into 'num_buckets' with no reordering. If no 
'num_buckets' is specified this will return the entire dimension in one bucket.
"""
import numpy as np
from .bucketer import Bucketer

class UniformRowBucketer(Bucketer):

    def extra_bytes_needed(self):
        return 0

    def name(self):
        return "uniform" + str(self.num_buckets)

    def bucket(self, X):
        assert (self.num_buckets <= X.shape[0])
        """ 
        Returns buckets and the original indexes of each row in each bucket.
        """
        row_bucket = []
        bucket_size = int(X.shape[0] / self.num_buckets) + 1
        bucket_remainder = X.shape[0] % self.num_buckets
        cur_start_index = 0
        for i in range(self.num_buckets):
            if i == bucket_remainder:
                bucket_size -= 1
            start, end = cur_start_index, cur_start_index + bucket_size
            end = X.shape[0] if end > X.shape[0] else end
            row_bucket.append(X[start:end, :])
            cur_start_index = end
        return row_bucket, np.arange(X.shape[0])


class UniformColBucketer:

    def name(self):
        return "uniform" + str(self.num_buckets)

    def extra_bytes_needed(self):
        return 0

    def bucket(self, row_buckets, X):
        assert (self.num_buckets <= X.shape[1])
        """ 
        Returns buckets and the original indexes of each column in each bucket.
        """
        final_buckets = []
        final_indexes = []
        col_indexes = None

        bucket_size_original = int(X.shape[1] / self.num_buckets) + 1
        bucket_remainder = X.shape[1] % self.num_buckets
        for bucket in row_buckets:
            col_bucket = []
            cur_start_index = 0
            bucket_size = bucket_size_original
            for i in range(self.num_buckets):
                if i == bucket_remainder:
                    bucket_size -= 1
                start, end = cur_start_index, cur_start_index + bucket_size
                end = X.shape[1] if end > X.shape[1] else end
                col_bucket.append(bucket[:, start:end])
                cur_start_index = end
            final_buckets.append(col_bucket)
            final_indexes.append(np.arange(X.shape[1]))
        return final_buckets, final_indexes
