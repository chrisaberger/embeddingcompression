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
    print("Embedding shape: " + str(embedding.shape))
    return vocab, embedding

def rowwise_kmeans(X, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters).fit(X)

    for i in tqdm(range(X.shape[0])):
        row = X[i, :]


filename = "../GloVe-1.2/vectors.txt"
load_embeddings(filename)
sorted_embedding = embedding#np.sort(embedding)
sort_indexes = np.argsort(embedding)

print(sorted_embedding)

median = np.median(sorted_embedding, axis=0)
mean = np.mean(sorted_embedding, axis=0)
max_vals = np.max(sorted_embedding, axis=0)
min_vals = np.min(sorted_embedding, axis=0)

middle = ((max_vals-min_vals)/2) + min_vals
sorted_embedding -= middle

frob_norm = np.linalg.norm(embedding)
one_d_embedding = embedding.flatten()

input_bytes = embedding.size*4
print("Frobenius Norm of Input: " + str(frob_norm))
print("Mean of Input: " + str(np.mean(one_d_embedding)))
print("Standard Deviation of Input: " + str(np.std(one_d_embedding)))
print("Input number of bytes: " + str(embedding.size*4))

print()

transposed = np.transpose(sorted_embedding)

codebook_size = n_clusters*transposed.shape[0]
print()
print("Num Clusters: " + str(codebook_size))

t0 = time.time()

compressed_embedding = np.zeros(transposed.shape)
for i in tqdm(range(transposed.shape[0])):
    row = transposed[i, :]
    kmeans = KMeans(n_clusters=n_clusters).fit(row.reshape(-1, 1))
    c_row = kmeans.cluster_centers_[kmeans.labels_] \
                                    .reshape(1, compressed_embedding.shape[1])
    compressed_embedding[i, :] = c_row

t1 = time.time()
print("KMeans time: " + str(t1-t0))

# to reconstruct 
compressed_embedding = np.transpose(compressed_embedding)
compressed_embedding += middle
x_inds_1d = np.arange(0, compressed_embedding.shape[0])
x_inds = np.zeros(compressed_embedding.shape, dtype=np.int32)
for i in range(0, compressed_embedding.shape[0]):
    x_inds[i, :] = x_inds_1d[i]
#compressed_embedding[x_inds, sort_indexes] = compressed_embedding

print()
print(compressed_embedding)
print()

embedding_indexes = (embedding.size*int(np.ceil(np.log2(n_clusters))))/8
codebook_bytes = codebook_size*4
compressed_bytes = codebook_bytes+embedding_indexes

print("Frobenius Norm of Compressed: " + str(np.linalg.norm(compressed_embedding)))
print("Mean of Compressed: " + str(np.mean(compressed_embedding)))
print("Standard Deviation of Compressed: " + str(np.std(compressed_embedding.flatten())))
print("Compressed Bytes: " + str(codebook_bytes+embedding_indexes))
print()
print("Frob norm diff: " + str(np.abs(np.linalg.norm(compressed_embedding)-frob_norm)))
print("Ratio: " + str(input_bytes/compressed_bytes))

pdv = vocab.to_frame()
pdv.rename(columns={0:'v'}, inplace=True)
result = pdv.join(pd.DataFrame(compressed_embedding))
result.to_csv("output.csv", sep=' ', index=False, header=False)
