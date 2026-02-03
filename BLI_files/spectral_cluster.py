import fasttext
import numpy as np
import os
from sklearn.cluster import MiniBatchKMeans

path_to_model = "/storage/ice-shared/vip-vyf/embeddings_team/models/array_models_3181969/model_9/model"
model = fasttext.load_model(path_to_model)

tokens = model.get_words()

embedding_list = []
for word in tokens:
    embedding_list.append(model.get_word_vector(word))
embedding_matrix = np.asarray(embedding_list, dtype=np.float32)
num_tokens, embedding_dim = embedding_matrix.shape

embedding_norms = np.linalg.norm(embedding_matrix, axis=1, keepdims=True)
embedding_norms = embedding_norms + 1e-12
normalized_embeddings = embedding_matrix / embedding_norms

num_clusters = 5000
num_landmarks = 5000
num_eig_dims = 200  # spectral dimension used for clustering only

random_seed = 118
random_state = np.random.RandomState(random_seed)
permutation_indices = random_state.permutation(num_tokens)
landmark_indices = permutation_indices[:num_landmarks]

landmark_embeddings = normalized_embeddings[landmark_indices]

similarity_matrix = landmark_embeddings @ landmark_embeddings.T
similarity_matrix[similarity_matrix < 0.0] = 0.0
similarity_matrix = similarity_matrix.astype(np.float64)

eigenvalues, eigenvectors = np.linalg.eigh(similarity_matrix)
sorted_indices = np.argsort(eigenvalues)[::-1][:num_eig_dims]

selected_eigenvalues = eigenvalues[sorted_indices].astype(np.float32)
selected_eigenvectors = eigenvectors[:, sorted_indices].astype(np.float32)

inverse_eigenvalues = 1.0 / (selected_eigenvalues + 1e-8)
scaled_eigenvectors = selected_eigenvectors * inverse_eigenvalues

cross_similarity = normalized_embeddings @ landmark_embeddings.T
cross_similarity[cross_similarity < 0.0] = 0.0
cross_similarity = cross_similarity.astype(np.float32)

spectral_embedding = cross_similarity @ scaled_eigenvectors
del cross_similarity

spectral_norms = np.linalg.norm(spectral_embedding, axis=1, keepdims=True)
spectral_norms = spectral_norms + 1e-12
normalized_spectral_embedding = spectral_embedding / spectral_norms
del spectral_embedding

kmeans = MiniBatchKMeans(
        n_clusters=num_clusters,
        batch_size=8192,
        n_init=5,
        max_iter=200,
        random_state=random_seed,
        )
cluster_labels = kmeans.fit_predict(normalized_spectral_embedding)
del normalized_spectral_embedding

centroids = []
for cluster_id in range(num_clusters):
    cluster_points = embedding_matrix[cluster_labels == cluster_id]
    if cluster_points.size == 0:
        centroids.append(None)
    else:
        centroid_vector = cluster_points.mean(axis=0)
        centroids.append(centroid_vector)

out_dir = "/home/hice1/mwarren64/vip-folder"
os.makedirs(out_dir, exist_ok=True)
centroid_path = os.path.join(out_dir, "centroids_300d.txt")

with open(centroid_path, "w", encoding="utf-8") as file:
    for centroid in centroids:
        if centroid is not None:
            file.write(" ".join(f"{value:.6f}" for value in centroid) + "\n")
