import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import time
import sys
from tqdm import tqdm
import os
import argparse

# Number of clusters per column.
n_clusters = 32


def load_embeddings(filename):
    """
    Loads a GloVe embedding at 'filename'. Returns a vector of strings that 
    represents the vocabulary and a 2-D numpy matrix that is the embeddings. 
    """
    df = pd.read_csv(
        filename,
        sep=' ',
        header=None,
        dtype={0: np.str},
        keep_default_na=False,
        na_values=[''])
    vocab = df[df.columns[0]]
    embedding = df.drop(df.columns[0], axis=1).as_matrix()
    print(embedding)
    print("Embedding shape: " + str(embedding.shape))
    return vocab, embedding


def bucket_rows(X, n_buckets):
    t0 = time.time()

    # Run kmeans to determine the buckets.
    kmeans = KMeans(n_clusters=n_buckets).fit(X)
    print("KMeans Bucket Rows Time: " + str(time.time() - t0))

    # Create the buckets and sort the vocabulary.
    buckets = {}
    index_buckets = {}    # original row index mapping
    new_V = []
    for i in range(len(kmeans.labels_)):
        label = kmeans.labels_[i]
        if label not in buckets:
            buckets[label] = [X[i, :]]
            index_buckets[label] = [i]
        else:
            buckets[label].append(X[i, :])
            index_buckets[label].append(i)
    """
    print(kmeans.cluster_centers_[0])
    for i in range(n_buckets):
        print(i)
        print(np.array(index_buckets[i]))
    """
    return buckets, index_buckets


def columnwise_kmean_compression(X, num_centroids_per_column):
    X_t = np.transpose(X)
    compressed_X_t = np.zeros(X_t.shape)
    for i in tqdm(range(X_t.shape[0])):
        row = X_t[i, :]
        kmeans = KMeans(n_clusters=num_centroids_per_column)\
                                            .fit(row.reshape(-1, 1))
        compressed_X_t[i, :] = kmeans.cluster_centers_[kmeans.labels_] \
                                        .reshape(1, compressed_X_t.shape[1])
    return np.transpose(compressed_X_t)


def print_stats(X, n_bytes, baseline_frob_norm=None):
    frob_norm = np.linalg.norm(X)
    one_d_X = X.flatten()
    print()
    print("Bytes Requried: " + str(n_bytes))
    print("Frobenius Norm of Input: " + str(frob_norm))
    print("Mean of Input: " + str(np.mean(one_d_X)))
    print("Standard Deviation of Input: " + str(np.std(one_d_X)))
    if baseline_frob_norm is not None:
        print("Frob norm diff: " + str(np.abs(frob_norm - baseline_frob_norm)))
    print()
    return frob_norm


def to_file(filename, V, X):
    if not os.path.exists("outputs"):
        os.makedirs("outputs")
    filename = os.path.join("outputs", filename)

    print(X)
    pdv = pd.DataFrame(V)
    pdv.rename(columns={0: 'v'}, inplace=True)
    result = pdv.join(pd.DataFrame(X))
    result.to_csv(filename, sep=' ', index=False, header=False)


def num_bits_needed(number):
    return np.ceil(np.log2(number))


def bucketed_columnwise_kmeans(vocab, embedding, num_buckets,
                               num_centroids_per_column):
    # Bucket method
    buckets, index_buckets = bucket_rows(embedding, num_buckets)
    compressed_embedding = None
    new_vocab = None
    for index in buckets:
        bucket = np.array(buckets[index])
        new_vals = columnwise_kmean_compression(bucket,
                                                num_centroids_per_column)
        new_vocab_words = vocab[np.array(index_buckets[index])].as_matrix()
        if compressed_embedding is None:
            compressed_embedding = new_vals
            new_vocab = new_vocab_words
        else:
            compressed_embedding = np.concatenate((compressed_embedding,
                                                   new_vals))
            new_vocab = np.concatenate((new_vocab, new_vocab_words))

    compressed_embedding = np.array(compressed_embedding)
    new_vocab = np.array(new_vocab)

    print(compressed_embedding.shape)
    compressed_embedding = np.array(compressed_embedding)

    # In each bucket there is a codebook for every column.
    num_codebooks = num_buckets * embedding.shape[1]
    # Codebook stores 'num_centroids_per_column' FP32 numbers.
    codebook_bytes = (32 * num_centroids_per_column) / 8
    # We need to store offsets to figure out where each bucket starts in the
    # matrix. Store (start_index, bucket_number) per bucket.
    # The bucket number actually only requires log2(num_buckets) bits. Store
    # the start_index as a 32 bit integer. Binary search in this structure
    # given an index to find what bucket you are in.
    codebook_offset_bytes = ((num_buckets + 1) *
                             (32 + num_bits_needed(num_buckets))) / 8
    total_codebook_bytes = (
        num_codebooks * codebook_bytes) + codebook_offset_bytes
    entry_bytes = (num_bits_needed(num_centroids_per_column
                                  ) * compressed_embedding.size) / 8

    total_bytes = entry_bytes + total_codebook_bytes
    print_stats(compressed_embedding, total_bytes, frob_norm)

    filename = "bucketwise_" + str(total_bytes) + "bytes_" \
                + str(num_buckets) + "buckets_" \
                + str(num_centroids_per_column) + "_centroids.txt"

    to_file(filename, new_vocab, compressed_embedding)


def columnwise_kmeans(vocab, embedding, num_centroids_per_column):
    compressed_embedding = columnwise_kmean_compression(embedding, 2)

    # Compute how much memory is needed.
    num_codebooks = embedding.shape[1]
    codebook_bytes = (32 * num_centroids_per_column) / 8
    total_codebook_bytes = num_codebooks * codebook_bytes
    entry_bytes = (num_bits_needed(num_centroids_per_column
                                  ) * compressed_embedding.size) / 8
    total_bytes = total_codebook_bytes + entry_bytes
    print_stats(compressed_embedding, total_bytes, frob_norm)

    filename = "columnwise_" + str(total_bytes) + "bytes_" + str(
        num_centroids_per_column) + "centroids.txt"
    to_file(filename, vocab.as_matrix(), compressed_embedding)


def rowwise_kmeans(vocab, X, num_centroids):
    t0 = time.time()

    # Run kmeans to determine the buckets.
    kmeans = KMeans(n_clusters=num_centroids).fit(X)
    print("KMeans Bucket Rows Time: " + str(time.time() - t0))

    compressed_X = kmeans.cluster_centers_[kmeans.labels_]
    print(compressed_X.shape)

    # Compute how much memory is needed.
    num_codebooks = num_centroids
    codebook_bytes = (32 * embedding.shape[1]) / 8
    total_codebook_bytes = num_codebooks * codebook_bytes
    entry_bytes = (num_bits_needed(num_centroids) * compressed_X.size) / 8
    total_bytes = total_codebook_bytes + entry_bytes
    print_stats(compressed_X, total_bytes, frob_norm)

    filename = "rowwise_" + str(total_bytes) + "bytes_" + str(
        num_centroids) + "centroids.txt"
    to_file(filename, vocab.as_matrix(), compressed_X)


def kmeans(vocab, X, num_centroids):
    t0 = time.time()

    # Run kmeans to determine the buckets.
    kmeans = KMeans(n_clusters=num_centroids).fit(X.reshape(-1,1))
    print("KMeans Bucket Rows Time: " + str(time.time() - t0))

    compressed_X = kmeans.cluster_centers_[kmeans.labels_].reshape(X.shape)
    print(compressed_X.shape)

    # Compute how much memory is needed.
    codebook_bytes = (32 * num_centroids) / 8
    entry_bytes = (num_bits_needed(num_centroids) * compressed_X.size) / 8
    total_bytes = codebook_bytes + entry_bytes
    print_stats(compressed_X, total_bytes, frob_norm)

    filename = "kmeans_" + str(total_bytes) + "bytes_" + str(
        num_centroids) + "centroids.txt"
    to_file(filename, vocab.as_matrix(), compressed_X)


def quantize(data, num_bits, scale_factor, biased=False):
    if not biased:
        random_data = np.random.uniform(0, 1, size=data.shape)
        data = np.floor((data / float(scale_factor)) + random_data)
    else:
        data = np.floor(data / float(scale_factor) + 0.5)
    min_value = -1 * (2**(num_bits - 1))
    max_value = 2**(num_bits - 1) - 1
    data = np.clip(data, min_value, max_value)
    return data * scale_factor


def naive_quantization(vocab, X, num_bits):
    print("naive quantization")

    min_val = np.amin(X)
    max_val = np.amax(X)

    center = (max_val-min_val)/2
    center = max_val-center

    X_recentered = X-center
    min_val = min_val-center
    max_val = max_val-center

    min_bit_value = -1 * (2**(num_bits - 1))
    max_bit_value = 2**(num_bits - 1) - 1

    sf_max = max_val/max_bit_value
    sf_min = min_val/min_bit_value

    sf = max(sf_min, sf_max)
    compressed_X = quantize(X, num_bits, sf)
    compressed_X += center

    total_bytes = (compressed_X.size*num_bits)/8
    print_stats(compressed_X, total_bytes, frob_norm)

    filename = "quant_" + str(total_bytes) + "bytes_" + str(num_bits)+"bits.txt"
    to_file(filename, vocab.as_matrix(), compressed_X)

parser = argparse.ArgumentParser()
parser.add_argument(
    "-f",
    "--filename",
    action="store",
    type=str,
    required=True,
    help="Path to input embeddngs file (in GloVe format).")
parser.add_argument(
    "--num_buckets",
    action="store",
    default=100,
    type=int,
    help="Number of buckets for bucketed columnwise kmeans.")
parser.add_argument(
    "-c",
    "--num_centroids",
    action="store",
    default=2,
    type=int,
    help="Number of centroids kmeans.")
parser.add_argument(
    "-b",
    "--num_bits",
    action="store",
    default=5,
    type=int,
    help="Number of bits for quantization.")
parser.add_argument(
    "--algo",
    action="store",
    default="all",
    type=str,
    choices=[
        "kmeans_bucketed_col", "kmeans_col", "kmeans_row", "kmeans", "quant",
        "quant_col", "quant_bucketed_col", "all"
    ],
    help="Solver/optimization algorithm.")
args = parser.parse_args()

vocab, embedding = load_embeddings(args.filename)
frob_norm = print_stats(embedding, embedding.size * 4)

if args.algo == "kmeans_bucketed_col" or args.algo == "all":
    print("Running bucketed columnwise kmeans...")
    bucketed_columnwise_kmeans(vocab, embedding, args.num_buckets,
                               args.num_centroids)
if args.algo == "kmeans_col" or args.algo == "all":
    print("Running columnwise kmeans...")
    columnwise_kmeans(vocab, embedding, args.num_centroids)
if args.algo == "kmeans_row" or args.algo == "all":
    print("Running rowwise kmeans...")
    rowwise_kmeans(vocab, embedding, args.num_centroids)
if args.algo == "kmeans" or args.algo == "all":
    print("Running kmeans quantization...")
    kmeans(vocab, embedding, args.num_centroids)
if args.algo == "quant" or args.algo == "all":
    print("Running naive quantization...")
    naive_quantization(vocab, embedding, args.num_bits)
