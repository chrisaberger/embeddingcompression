import numpy as np
from .quantizer import Quantizer
from sklearn.cluster import KMeans



class LloydMaxQuantizer(Quantizer):

    def __init__(self, num_centroids, q_dim_row, q_dim_col):
        self.num_centroids = num_centroids
        self.q_dim_row = q_dim_row # num rows to quant together
        self.q_dim_col = q_dim_col # num cols to quant together

    def name(self):
        return "lloydmax" + str(self.num_centroids) + "b"

    def _check_dims(bucket_shape_row, bucket_shape_col):
        if bucket_shape_row % self.q_dim_row != 0:
            raise ValueError("bucket and quantization row dimension mismatch")
        if bucket_shape_col % self.q_dim_cols != 0:
            raise ValueError("bucket and quantization col dimension mismatch")
        return 0
            
    def quantize(self, X):
        #make sure the dims check out
        num_rows_in_buck = X.shape[0]
        num_cols_in_buck = X.shape[1]
        _check_dims(num_rows_in_buck)

        #double loop over quantized submats in the bucket
            #across "horiztionally" first, then down "vertically"
    
        points = []
        for i in range(0,
        









class KmeansQuantizer(Quantizer):

    def __init__(self, num_centroids):
        self.num_centroids = num_centroids

    def name(self):
        return "kmeans" + str(self.num_centroids) + "b"

    def quantize(self, X):
        kmeans = KMeans(n_clusters=self.num_centroids,max_iter=120,tol=0.01)\
                                            .fit(X.reshape(-1, 1))
        compressed_X = kmeans.cluster_centers_[kmeans.labels_] \
                                        .reshape(X.shape)

        codebook_bytes = self.num_centroids * 4
        entry_bytes = (
            np.ceil(np.log2(self.num_centroids)) * compressed_X.size) / 8
        total_bytes = codebook_bytes + entry_bytes

        return compressed_X, total_bytes
