import fasttext
import numpy as np

path_to_model = "/storage/ice-shared/vip-vyf/embeddings_team/models/array_models_2947993/model_1/model"
model = fasttext.load_model(path_to_model)

input_path = "/home/hice1/mwarren64/vip-folder/centroids.txt"
output_path = "/home/hice1/mwarren64/vip-folder/centroids_neighbors.txt"

words = model.get_words()
word_vecs = np.vstack([model.get_word_vector(w) for w in words])
word_vecs /= np.linalg.norm(word_vecs, axis=1, keepdims=True) + 1e-12  # normalize

rows = []
with open(input_path, "r", encoding="utf-8") as f:
    for line in f:
        vec = np.fromstring(line.strip(), sep=" ")
        vec /= np.linalg.norm(vec) + 1e-12
        sims = word_vecs @ vec  # cosine similarities
        top_idx = np.argsort(-sims)[:5]
        row = [words[i] for i in top_idx]
        rows.append(row)

with open(output_path, "w", encoding="utf-8") as f:
    for row in rows:
        f.write(",".join(row) + "\n")

