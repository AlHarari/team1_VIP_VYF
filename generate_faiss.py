import fasttext
import faiss
import numpy as np
import csv
import os

# Set paths.
common_path = "/storage/ice-shared/vip-vyf/embeddings_team/models/array_models_3181969/model_9/" # Change to path of the model you're interested in.
model_path = common_path + "model"
matrix_path = common_path + "input_matrix_model9.npy"

# Get models and matrix
model = fasttext.load_model(model_path)
words = model.get_words() # In the future, we'll probably just read the words_list.tsv file.
input_matrix = np.load(matrix_path)
input_matrix = input_matrix / np.linalg.norm(input_matrix, axis=1, keepdims=True)
ROW, DIM = input_matrix.shape

# Build the index.
idx = faiss.IndexFlatIP(DIM)
idx.add(input_matrix)

# helper function to query nearest neighbors
def get_neighbors(word, k):
    query_vec = model[word].reshape((1, DIM)) # This is why we can't query multiple vectors at one, unfortunately.
    query_vec = query_vec / np.linalg.norm(query_vec, axis=1, keepdims=True)
    sims, neighbors_idx = idx.search(query_vec, k)
    neighbors = [(words[i], sims[0][j]) for j, i in enumerate(neighbors_idx[0])]
    return neighbors

greek_words = [
    "τρέξιμο",      # run
    "σκούντημα",        # jog
    "βόλτα",     # walk
    "άλμα",  # jump
    "αγάπη",     # love
    "κατάρα",     # curse
    "επιτυχία",     # hit
    "μιλώ",   # speak
]

save_dir = "./corpus_stuff"
os.makedirs(save_dir, exist_ok=True)

csv_path = os.path.join(save_dir, "greek_neighbors_normalized.csv")

with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Query", "Neighbor_Native", "Similarity_Native", "Neighbor_Faiss", "Similarity_Faiss", ])
    for qw in greek_words:
        results_faiss = get_neighbors(qw, k=10)
        results_native = model.get_nearest_neighbors(qw, k=10)
        for (neigh_f, score_f), (score_n, neigh_n) in zip(results_faiss, results_native):
            writer.writerow([qw, neigh_n, score_n, neigh_f, score_f])
