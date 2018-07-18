import numpy as np
from .quantizer import Quantizer
from sklearn.cluster import KMeans


class LloydMaxQuantizer(Quantizer):
    def __init__(self, num_centroids, q_dim_row, q_dim_col):
        self.num_centroids = num_centroids
        self.q_dim_row = q_dim_row  # num rows to quant together
        self.q_dim_col = q_dim_col  # num cols to quant together

    def name(self):
        return "lloydmax" + str(self.num_centroids) + "b"

    def _check_dims(self, bucket_shape_row, bucket_shape_col):
        if bucket_shape_row % self.q_dim_row != 0:
            raise ValueError("bucket and quantization row dimension mismatch")
        if bucket_shape_col % self.q_dim_col != 0:
            raise ValueError("bucket and quantization col dimension mismatch")
        return 0

    def quantize(self, X):
        #make sure the dims check out
        num_rows_in_buck = X.shape[0]
        num_cols_in_buck = X.shape[1]
        self._check_dims(num_rows_in_buck, num_cols_in_buck)

        #double loop over quantized submats in the bucket
        #across "horizontally" first, then down "vertically"
        points = []
        for i in range(0, num_rows_in_buck, self.q_dim_row):
            for j in range(0, num_cols_in_buck, self.q_dim_col):
                i_prime = i + self.q_dim_row
                j_prime = j + self.q_dim_col
                points.append(X[i:i_prime, j:j_prime])

        print(len(points))
        #now we can reshape all points for k-means
        quant_dim = self.q_dim_row * self.q_dim_col
        vectorized_points = [p.reshape(1, quant_dim) for p in points]
        stacked_vectorized_pts = np.vstack(vectorized_points)
        print(stacked_vectorized_pts.shape)

        #now that we have our sample points, we can go ahead and run k-means
        kmeans = KMeans(n_clusters=self.num_centroids,max_iter=120,tol=0.01)\
                                   .fit(stacked_vectorized_pts)
        compressed_X = np.zeros([num_rows_in_buck, num_cols_in_buck])

        #we can inflated the k-means into the compressed_X mat
        c = 0  # counter for centroids
        for i in range(0, num_rows_in_buck, self.q_dim_row):
            for j in range(0, num_cols_in_buck, self.q_dim_col):
                centroid = kmeans.cluster_centers_[kmeans.labels_[c]]
                centroid = centroid.reshape(self.q_dim_row, self.q_dim_col)
                c += 1
                i_prime = i + self.q_dim_row
                j_prime = j + self.q_dim_col
                compressed_X[i:i_prime, j:j_prime] = centroid

        #finally, tally up the bytes
        codebook_bytes = self.num_centroids * 4 * quant_dim
        entry_bytes = (np.ceil(np.log2(self.num_centroids)) * compressed_X.size
                       / quant_dim) / 8
        total_bytes = codebook_bytes + entry_bytes

        return compressed_X, total_bytes


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
