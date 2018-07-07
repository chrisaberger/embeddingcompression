import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import time
import sys
from tqdm import tqdm

# Number of clusters per column.
n_clusters = 32

def load_embeddings(filename):
    """
    Loads a GloVe embedding at 'filename'. Returns a vector of strings that 
    represents the vocabulary and a 2-D numpy matrix that is the embeddings. 
    """
    df = pd.read_csv( filename, 
                      sep=' ', 
                      header=None, 
                      dtype= {0:np.str}, 
                      keep_default_na=False, 
                      na_values=[''] )
    vocab = df[df.columns[0]]
    embedding = df.drop(df.columns[0], axis=1).as_matrix()
    print(embedding)
    print("Embedding shape: " + str(embedding.shape))
    return vocab, embedding

def bucket_rows(X, n_buckets):
    t0 = time.time()

    # Run kmeans to determine the buckets.
    kmeans = KMeans(n_clusters=n_buckets).fit(X)
    print("KMeans Bucket Rows Time: " + str(time.time()-t0))

    # Create the buckets and sort the vocabulary.    
    buckets = {}
    index_buckets = {} # original row index mapping
    new_V = []
    for i in range(len(kmeans.labels_)):
        label = kmeans.labels_[i]
        if label not in buckets:
            buckets[label] = [X[i,:]]
            index_buckets[label] = [i]
        else:
            buckets[label].append(X[i,:])
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

def print_stats(X, baseline_frob_norm = None):
    frob_norm = np.linalg.norm(X)
    one_d_X = X.flatten()
    print()
    print("Frobenius Norm of Input: " + str(frob_norm))
    print("Mean of Input: " + str(np.mean(one_d_X)))
    print("Standard Deviation of Input: " + str(np.std(one_d_X)))
    if baseline_frob_norm is not None:
        print("Frob norm diff: " + str(np.abs( frob_norm - baseline_frob_norm )))
    print()
    return frob_norm

def to_file(filename, V, X):
    print(X)
    pdv = pd.DataFrame(V)
    pdv.rename(columns={0:'v'}, inplace=True)
    result = pdv.join(pd.DataFrame(X))
    result.to_csv(filename, sep=' ', index=False, header=False)

filename = "../GloVe-1.2/vectors.txt"
vocab, embedding = load_embeddings(filename)
frob_norm = print_stats(embedding)

# Bucket method
buckets, index_buckets = bucket_rows(embedding, 200)
compressed_embedding = None
new_vocab = None
for index in buckets:
    bucket = np.array(buckets[index])
    new_vals = columnwise_kmean_compression(bucket, 2)
    if compressed_embedding is None:
        compressed_embedding = new_vals
        new_vocab = vocab[np.array(index_buckets[index])].as_matrix()
    else:
        compressed_embedding = np.concatenate((compressed_embedding, new_vals))
        new_vocab = np.concatenate((new_vocab, vocab[np.array(index_buckets[index])].as_matrix()))

compressed_embedding = np.array(compressed_embedding)
new_vocab = np.array(new_vocab)

print(compressed_embedding.shape)
compressed_embedding = np.array(compressed_embedding)
print_stats(compressed_embedding, frob_norm)
to_file("bucketwise.txt", new_vocab, compressed_embedding)
# end bucket method

# Columnwise method
compressed_embedding = columnwise_kmean_compression(embedding, 2)
print_stats(compressed_embedding, frob_norm)
to_file("columnwise.txt", vocab.as_matrix(), compressed_embedding)
# end columnwise method
