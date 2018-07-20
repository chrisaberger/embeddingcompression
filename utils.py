import numpy as np
import pandas as pd
import os
import argparse
import csv
import sys


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
        "-o",
        "--output_folder",
        action="store",
        default="outputs",
        type=str,
        help="Folder to output embeddings to.")
    parser.add_argument(
        "--row_bucketer",
        action="store",
        default="uniform",
        type=str,
        choices=["uniform", "kmeans", "sorted"],
        help="Row bucketing strategy.")
    parser.add_argument(
        "--col_bucketer",
        action="store",
        default="uniform",
        type=str,
        choices=["uniform", "kmeans", "sorted"],
        help="Column bucketing strategy.")
    parser.add_argument(
        "--quantizer",
        action="store",
        default="uniform_fp",
        type=str,
        choices=["uniform_fp", "kmeans", "uniform_mt"],
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
    parser.add_argument(
        "--quant_num_rows",
        action="store",
        default=1,
        type=int,
        help="Number of rows in a vector/matrix Lloyd-Max quantization.")
    parser.add_argument(
        "--quant_num_cols",
        action="store",
        default=1,
        type=int,
        help="Number of cols in a vector/matrix Lloyd-Max quantization.")

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

    file = open(filename, "r") 
    lines = file.readlines()
    vocab = []
    embedding = []
    for line in lines:
        values = line.strip("\n").split(" ")
        vocab.append(values.pop(0))
        embedding.append([float(v) for v in values])
    embedding = np.array(embedding)
    vocab = np.array(vocab)
    file.close()

    print("Embedding shape: " + str(embedding.shape))
    return vocab, embedding


def print_stats(X, q_X, n_bytes):
    print()
    print("Bytes Requried: " + str(n_bytes))
    print("Compression Ratio: " + str((X.size * 4) / n_bytes))
    print("Frobenius Norm of Original Matrix: " + str(np.linalg.norm(X)))
    print("Frobenius Norm of Compressed Matrix: " + str(np.linalg.norm(q_X)))
    print()


def create_filename(output_folder, input_filename, row_bucketer, col_bucketer,
                    quantizer, num_bytes, q_d_r, q_c_r):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_folder = os.path.join(output_folder,
                                 os.path.basename(input_filename))
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    filename = "q" + quantizer.name()\
            + "_d" + str(q_d_r) + "_" + str(q_c_r)\
            + "_r" + row_bucketer.name()\
            + "_c" + col_bucketer.name()\
            + "_bytes" + str(num_bytes) + ".txt"
    return os.path.join(output_folder, filename)


def to_file(filename, V, X):

    file = open(filename, "w") 
    for i in range(V.size):
        file.write(V[i] + " ")
        row = X[i, :]
        strrow = [str(r) for r in row]
        file.write(" ".join(strrow))
        file.write("\n")
    file.close()

def num_bits_needed(number):
    return np.ceil(np.log2(number))
