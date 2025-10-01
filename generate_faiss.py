import fasttext
import faiss
import numpy as np
import csv
import os

# Create matrix from vectors list file.
model_path = "/storage/ice-shared/vip-vyf/embeddings_team/models/array_models_3181969/model_9/model"
model = fasttext.load_model(model_path)
words = model.get_words() # In the future, we'll probably just read the words_list.tsv file.
input_matrix = np.load("./corpus_stuff/input_matrix_model9.npy") # Need to find a way to ensure that the model correponds to the input matrix!
ROW, DIM = input_matrix.shape

# Build the index.
idx = faiss.IndexFlatIP(DIM)
idx.add(input_matrix)

# helper function to query nearest neighbors
def get_neighbors(word, k):
    query_vec = model[word].reshape((1, DIM))
    sims, neighbors_idx = idx.search(query_vec, k)
    neighbors = [(words[i], sims[0][j]) for j, i in enumerate(neighbors_idx[0])] # In the future, it is possible that we'll query for multiple vectors at a time
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

csv_path = os.path.join(save_dir, "greek_neighbors_1.csv")

with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Query", "Neighbor_Native", "Similarity_Native", "Neighbor_Faiss", "Similarity_Faiss", ])
    for qw in greek_words:
        results_faiss = get_neighbors(qw, k=10)
        results_native = model.get_nearest_neighbors(qw, k=10)
        for (neigh_f, score_f), (score_n, neigh_n) in zip(results_faiss, results_native):
            writer.writerow([qw, neigh_n, score_n, neigh_f, score_f])