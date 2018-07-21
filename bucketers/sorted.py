"""
Uniformly chunks a matrix into 'num_buckets' with no reordering. If no 
'num_buckets' is specified this will return the entire dimension in one bucket.
"""
import numpy as np
from .bucketer import Bucketer
from . import UniformRowBucketer


class SortedBucketer(UniformRowBucketer):
    def __init__(self, num_buckets, max_num_buckets):
        Bucketer.__init__(self, num_buckets, max_num_buckets)

    def name(self):
        return "sorted" + str(self.num_buckets)

    def bucket(self, X):
        assert (self.num_buckets <= X.shape[0])
        """
        A thought experiment -- sorts without regard to matrix structure
        """

        #First, perform the orderless sort
        vectorized_X = X.reshape(-1, 1)
        sort_vect_X = np.sort(vectorized_X, axis=0)
        argsort_vect_X = np.argsort(vectorized_X, axis=0)
        X_sorted = sort_vect_X.reshape(X.shape)

        #Now, let me be lazy and use an internal bucketer to perform fixed bucketing
        internal_bucketer = UniformRowBucketer(self.num_buckets,
                                               self.max_num_buckets)
        row_buckets, row_reorder = internal_bucketer.bucket(X_sorted)
        row_reorder = [row_reorder, argsort_vect_X]
        return row_buckets, row_reorder
