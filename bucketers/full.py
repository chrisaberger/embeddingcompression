import numpy as np

class FullRowBucketer:
    def name(self):
        return "full"

    def bucket(self, X):
        return [X], np.arange(X.shape[0])

class FullColBucketer:
    def name(self):
        return "full"

    def extra_bytes_needed(self):
        return 0

    def bucket(self, row_buckets, X):
        return [row_buckets], [[np.arange(X.shape[1])]]