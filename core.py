from quantizers import uniform
import utils
import numpy as np
from tqdm import tqdm

def bucket(row_bucketer, col_bucketer, X):
    """
    We return a list of lists of buckets of data (the other two things specify
    (1) how to get back the rows and (2) how to get back the cols within each 
    bucket). The first list represents the row
    buckets. The second list represents the column buckets.  If you need to 
    do the column wise method first just take the transpose upon entrance and
    exit of this method.
    """

    row_buckets, row_reorder = row_bucketer.bucket(X)
    buckets, col_reorder = col_bucketer.bucket(row_buckets, X)
    return buckets, row_reorder, col_reorder


def quantize(buckets, quantizer):
    """
    Quantizes each bucket!
    """
    q_buckets = []
    num_bytes = 0
    print("Quantizing...")
    for row_bucket in tqdm(buckets):
        q_col_buckets = []
        for col_bucket in row_bucket:
            q_X, q_bytes = quantizer.quantize(col_bucket)
            num_bytes += q_bytes
            q_col_buckets.append(q_X)
        q_buckets.append(q_col_buckets)
    return q_buckets, num_bytes


def finish(buckets, num_bytes, X, V, row_reorder, col_reorder, filename):
    """
    Reorders everything back, prints some stats, and sends the compressed 
    embeddings to a file in GloVe format. The output files always appear in the
    'output' directory.
    """
    compressed_X = None
    print("Reconstructing...")
    for i in tqdm(range(len(buckets))):
        reconstructed_bucket = None
        for j in range(len(buckets[i])):
            # sort back by col_reorder index
            # stitch into large 'reconstructed' matrix.
            if reconstructed_bucket is None:
                reconstructed_bucket = buckets[i][j]
            else:
                reconstructed_bucket = np.append(reconstructed_bucket, buckets[i][j], axis = 1)

        if compressed_X is None:
            compressed_X = reconstructed_bucket
        else:
            compressed_X = np.concatenate((compressed_X,
                                           reconstructed_bucket))

    if np.array_equal(col_reorder, np.arange(X.shape[1])):
        # This is extremely slow but writing the for loop is not.
        #compressed_X = compressed_X[:, col_reorder]
        new_compressed_X = None
        for c in tqdm(col_reorder):
            if new_compressed_X is None:
                new_compressed_X = compressed_X[:, c]
            else:
                np.concatenate((new_compressed_X, compressed_X[:, c]))
        compressed_X = new_compressed_X

    # reorder the 'reconstructed' matrix. by row reorder
    print("Reodering rows...")
    compressed_X = compressed_X[row_reorder, :]

    # Print stats and send to file.
    utils.print_stats(X, compressed_X, num_bytes)

    utils.to_file(filename, V.as_matrix(), compressed_X)
