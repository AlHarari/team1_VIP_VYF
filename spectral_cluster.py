import fasttext
import numpy as np
import os
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import coo_matrix, diags, eye
from scipy.sparse.linalg import eigsh

path_to_model = "/storage/ice-shared/vip-vyf/embeddings_team/models/array_models_2947993/model_1/model"
model = fasttext.load_model(path_to_model)

tokens = model.get_words()
vectors = [model.get_word_vector(w) for w in tokens]
x = np.array(vectors)
n, dim = x.shape

k_nn = 50
nbrs = NearestNeighbors(n_neighbors=k_nn + 1, metric="cosine", n_jobs=-1)
nbrs.fit(x)
distances, indices = nbrs.kneighbors(x)
sims = 1.0 - distances

row_idx, col_idx, data = [], [], []
for i in range(n):
    for j in range(1, k_nn + 1):
        sim = sims[i, j]
        if sim > 0:
            row_idx.append(i)
            col_idx.append(indices[i, j])
            data.append(sim)

A = coo_matrix((data, (row_idx, col_idx)), shape=(n, n)).tocsr()
A = A.maximum(A.transpose()).tocsr()

deg = np.array(A.sum(axis=1)).ravel()
D_inv_sqrt = diags(1.0 / np.sqrt(deg + 1e-12))
L = eye(n, format="csr") - D_inv_sqrt @ A @ D_inv_sqrt

n_clusters = 500
eig_vals, eig_vecs = eigsh(L, k=n_clusters, which="SM")
U = eig_vecs / (np.linalg.norm(eig_vecs, axis=1, keepdims=True) + 1e-12)
labels = np.argmax(U, axis=1)

centroids, members = [], []
for cid in range(n_clusters):
    idx = np.where(labels == cid)[0]
    if len(idx) == 0:
        centroids.append(None)
        members.append([])
        continue
    centroids.append(x[idx].mean(axis=0))
    members.append(idx)

out_dir = "/home/hice1/mwarren64/vip-folder"
os.makedirs(out_dir, exist_ok=True)
centroid_path = os.path.join(out_dir, "centroids.txt")

with open(centroid_path, "w", encoding="utf-8") as f:
    for c in centroids:
        if c is not None:
            f.write(" ".join(f"{v:.6f}" for v in c) + "\n")

