from quantizers import uniform
import utils
import numpy as np

def bucket(row_bucketer, col_bucketer, X):
    """
    Returns a list of buckets. A reordered vocab. A way to rearrange the columns
    in each bucket. 
    """

    # bucket the rows first, if reordered reoder the vocab.

    # Inside of each bucket perform the column bucketing. If the column
    # bucketing causes a reorder store the index map for each column.

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
    q_buckets = []
    num_bytes = 0
    for row_bucket in buckets:
        q_col_buckets = []
        for col_bucket in row_bucket:
            q_X, q_bytes = quantizer.quantize(col_bucket)
            num_bytes += q_bytes
            q_col_buckets.append(q_X)
        q_buckets.append(q_col_buckets)

    return q_buckets, num_bytes

def finish(buckets, num_bytes, X, V, row_reorder, col_reorder, filename):
    # Reorder everything back, print some stats, and send to file.
    compressed_X = None
    for i in range(len(buckets)):
        for j in range(len(buckets[i])):
            # sort back by col_reorder index
            # stitch into large 'reconstructed' matrix.
            bucket = buckets[i][j]
            col_ordering = col_reorder[i][j]

            if compressed_X is None:
                compressed_X = bucket[:,col_ordering]
            else:
                compressed_X = np.concat(compressed_X, bucket[:,col_ordering])

    # reorder the 'reconstructed' matrix. by row reorder
    compressed_X = compressed_X[row_reorder, :]

    # Print stats and send to file.
    utils.print_stats(compressed_X, num_bytes)

    utils.to_file(filename, V.as_matrix(), compressed_X)