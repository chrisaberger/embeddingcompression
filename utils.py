import numpy as np
import pandas as pd
import os

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

def print_stats(X, n_bytes, baseline_frob_norm=None):
    frob_norm = np.linalg.norm(X)
    one_d_X = X.flatten()
    print()
    print("Bytes Requried: " + str(n_bytes))
    print("Frobenius Norm of Matrix: " + str(frob_norm))
    print("Mean of Matrix: " + str(np.mean(one_d_X)))
    print("Standard Deviation of Matrix: " + str(np.std(one_d_X)))
    if baseline_frob_norm is not None:
        print("Frob norm diff: " + str(np.abs(frob_norm - baseline_frob_norm)))
    print()
    return frob_norm

def get_filename(row_bucketer, col_bucketer, quantizer, num_bytes):
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
