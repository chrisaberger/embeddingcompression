import numpy as np
from .quantizer import Quantizer 
from sklearn.cluster import KMeans

class KmeansQuantizer(Quantizer):
    def __init__(self, num_centroids):
        self.num_centroids = num_centroids

    def name(self):
        return "kmeans" + str(self.num_centroids) + "b"

    def quantize(self, X):

        kmeans = KMeans(n_clusters=self.num_centroids)\
                                            .fit(X.reshape(-1, 1))
        compressed_X = kmeans.cluster_centers_[kmeans.labels_] \
                                        .reshape(X.shape)

        codebook_bytes = self.num_centroids * 4
        entry_bytes = (np.ceil(np.log2(self.num_centroids)) * compressed_X.size) / 8
        total_bytes = codebook_bytes + entry_bytes

        return compressed_X, total_bytes
