import numpy as np
import pandas as pd
import os
import argparse
import configparser


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--filename",
        action="store",
        type=str,
        required=True,
        help="Path to input embeddngs file (in GloVe format).")
    parser.add_argument(
        "--row_bucketer",
        action="store",
        default="uniform",
        type=str,
        choices=["uniform", "kmeans"],
        help="Row bucketing strategy.")
    parser.add_argument(
        "--col_bucketer",
        action="store",
        default="uniform",
        type=str,
        choices=["uniform", "kmeans"],
        help="Column bucketing strategy.")
    parser.add_argument(
        "--quantizer",
        action="store",
        default="uniform",
        type=str,
        choices=["uniform", "kmeans"],
        help="Quantization strategy.")
    parser.add_argument(
        "--num_row_buckets",
        action="store",
        default=1,
        type=int,
        help="Number of row buckets.")
    parser.add_argument(
        "--num_col_buckets",
        action="store",
        default=1,
        type=int,
        help="Number of col buckets.")
    parser.add_argument(
        "--num_bits",
        action="store",
        default=5,
        type=int,
        help="Number of bits for uniform quantization.")
    parser.add_argument(
        "--num_centroids",
        action="store",
        default=2,
        type=int,
        help="Number of centroids for kmeans quantization.")
    args = parser.parse_args()
    print(args)
    return args


def load_config(filename):
    config = configparser.ConfigParser()
    config.read(filename)
    return config


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

    print("Embedding shape: " + str(embedding.shape))
    return vocab, embedding


def print_stats(X, q_X, n_bytes):
    print()
    print("Bytes Requried: " + str(n_bytes))
    print("Compression Ratio: " + str((X.size * 4) / n_bytes))
    print("Frobenius Norm of Original Matrix: " + str(np.linalg.norm(X)))
    print("Frobenius Norm of Compressed Matrix: " + str(np.linalg.norm(q_X)))
    print()


def create_filename(row_bucketer, col_bucketer, quantizer, num_bytes):
    return "q" + quantizer.name()\
            + "_r" + row_bucketer.name()\
            + "_c" + col_bucketer.name()\
            + "_bytes" + str(num_bytes) + ".txt"


def to_file(filename, V, X):
    if not os.path.exists("outputs"):
        os.makedirs("outputs")
    filename = os.path.join("outputs", filename)

    pdv = pd.DataFrame(V)
    pdv.rename(columns={0: 'v'}, inplace=True)
    result = pdv.join(pd.DataFrame(X))
    result.to_csv(filename, sep=' ', index=False, header=False)


def num_bits_needed(number):
    return np.ceil(np.log2(number))
