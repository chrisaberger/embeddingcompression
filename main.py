import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import time

"""
Step 1: Parse a GloVe file, return both a vocab vector full of strings and a 
embeddings matrix full of the embeddings.
"""
filename = "../GloVe-1.2/vectors.txt"
df = pd.read_csv(filename, sep=' ', header=None)
vocab = df.iloc[:,0].as_matrix()
embedding = df.drop(df.columns[0], axis=1).as_matrix()

sorted_embedding = np.sort(embedding)
print(sorted_embedding)
#print(sorted_embedding.shape)
#print(vocab)

median = np.median(sorted_embedding, axis=0)
mean = np.mean(sorted_embedding, axis=0)
max_vals = np.max(sorted_embedding, axis=0)
min_vals = np.min(sorted_embedding, axis=0)

middle = ((max_vals-min_vals)/2) + min_vals
sorted_embedding -= middle

print(sorted_embedding)
frob_norm = np.linalg.norm(embedding)
one_d_embedding = np.sort(embedding.flatten())

print("Frobenius Norm of Input: " + str(frob_norm))
print("Mean of Input: " + str(np.mean(one_d_embedding)))
print("Standard Deviation: " + str(np.std(one_d_embedding)))
print()

k = 100
n_clusters = int(sorted_embedding.shape[0]/k)
print("Num Clusters: " + str(n_clusters))
print()
t0 = time.time()
kmeans = KMeans(n_clusters=n_clusters).fit(sorted_embedding)
compressed_embedding = []
for i in range(0, len(kmeans.labels_)):
    compressed_embedding.append(sorted_embedding[kmeans.labels_[i],:])
compressed_embedding = np.array(compressed_embedding)
compressed_embedding += max_vals
t1 = time.time()

print("Frobenius Norm of Input: " + str(np.linalg.norm(compressed_embedding)))
print("Mean of Input: " + str(np.mean(compressed_embedding)))
print("Standard Deviation: " + str(np.std(compressed_embedding.flatten())))
np.save(file="compressed.npy", arr=compressed_embedding)



