import numpy as np
from scipy.sparse import csr_matrix
from . import Quantizer


class PruneQuantizer(Quantizer):

    def __init__(self, prune_percentage):
        self.pp = prune_percentage

    def name(self):
        return "prune" + str(self.pp) + "b"

    def get_total_bytes(self, X_sparse):
        X_csr = csr_matrix(X_sparse, dtype='float16')
        return X_csr.data.nbytes + X_csr.indptr.nbytes + X_csr.indices.nbytes

    def quantize(self, X):
        pruned_X = np.copy(X)
        cutoff = np.percentile(np.abs(pruned_X.flatten()), self.pp)
        pruned_X[np.abs(pruned_X) < cutoff] = 0
        return pruned_X, self.get_total_bytes(pruned_X)


class CrazyPruneQuantizer(PruneQuantizer):

    def name(self):
        return "crazyprune" + str(self.pp) + "b"

    def quantize(self, X):
        pruned_X = np.copy(X)
        cutoff = np.percentile(np.abs(pruned_X.flatten()), self.pp)
        pruned_X[np.abs(pruned_X) < cutoff] = 0
        X_pruned = pruned_X
        X_pruned[X_pruned > 0.1] = 0.5
        X_pruned[X_pruned < -0.1] = -0.5
        return X_pruned, self.get_total_bytes(X_pruned)

