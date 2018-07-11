from quantizers import uniform
import utils
import numpy as np
from tqdm import tqdm


def bucket(row_bucketer, col_bucketer, X):
    """
    We return a list of lists of buckets of data  The first list represents the 
    row buckets. The second list represents the column buckets.  If you need to 
    do the column wise method first just take the transpose upon entrance and
    exit of this method. The other two returned arrays specify the mapping from
    (1) the bucketed rows back to original indexes (in order) and (2) the 
    bucketed columns back to the original indexes (in order).
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
    print("Reconstructing...")

    # Allocate buffers for each row bucket. These are used when the columns are
    # reconstructed.
    reconstruction_buffers = []
    for i in range(len(buckets)):
        assert (len(buckets[i]) > 0)
        reconstruction_buffers.append(
            np.zeros([buckets[i][0].shape[0], X.shape[1]]))
    # Buffer for the final output.
    compressed_X = np.zeros(X.shape)

    # The indexes into 'compressed_X' for the rows.
    row_start = 0
    row_end = 0
    for i in tqdm(range(len(buckets))):
        reconstructed_bucket = None
        # all column buckets have the same number of rows.
        row_end = row_start + buckets[i][0].shape[0]

        # The buffer we will reorganize the columns back into.
        reconstructed_bucket = reconstruction_buffers[i]

        # The col indexes for the 'reconstructed_bucket'.
        col_start = 0
        col_end = 0

        for j in range(len(buckets[i])):
            # stitch into large 'reconstructed' matrix.
            col_end = col_start + buckets[i][j].shape[1]
            reconstructed_bucket[:, col_start:col_end] = buckets[i][j]
            col_start = col_end

        # Sort the buffer by the 'col_reorder' index.
        new_x = np.zeros(reconstructed_bucket.shape)
        for c_i in range(len(col_reorder[i])):
            new_x[:, col_reorder[i][c_i]] = reconstructed_bucket[:, c_i]

        # Place it into our final buffer after it is sorted.
        compressed_X[row_start:row_end, :] = new_x

        row_start = row_end

    print(compressed_X.shape)

    # reorder the 'reconstructed' matrix. by row reorder
    new_x = np.zeros(compressed_X.shape)
    for i in range(compressed_X.shape[0]):
        new_x[row_reorder[i], :] = compressed_X[i, :]
    #compressed_X = compressed_X[row_reorder, :]
    compressed_X = np.copy(new_x, order="F")

    # Print stats and send to file.
    utils.print_stats(X, compressed_X, num_bytes)

    utils.to_file(filename, V.as_matrix(), compressed_X)
