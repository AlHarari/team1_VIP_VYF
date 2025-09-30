import fasttext
import faiss
import numpy as np
import csv
import os

# This model performed the best so far
path_to_model = "/storage/ice-shared/vip-vyf/embeddings_team/models/array_models_2947993/model_1/model"
model = fasttext.load_model(path_to_model)
input_matrix = model.get_input_matrix()
words = model.get_words()

rows, cols = input_matrix.shape

# Build the index.
idx = faiss.IndexFlatIP(cols)
idx.add(input_matrix)

# helper function to query nearest neighbors
def get_neighbors(word, k):
    if word not in words:
        print(f"'{word}' not in corpus")
        return []
    word_idx = words.index(word)
    query_vec = input_matrix[word_idx].reshape(1, cols)
    sims, neighbors_idx = idx.search(query_vec, k)
    neighbors = [(words[i], sims[0][j]) for j, i in enumerate(neighbors_idx[0])]
    return neighbors
``
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

save_dir = "/storage/ice-shared/vip-vyf/embeddings_team/models/array_models_2947993/model_1/"
os.makedirs(save_dir, exist_ok=True)

csv_path = os.path.join(save_dir, "greek_neighbors_faiss_1.csv")

with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Query", "Neighbor", "Similarity"])
    for qw in greek_words:
        results = get_neighbors(qw, k=10)
        if not results:
            print(f"warning '{qw}' not in vocab")
        for neighbor, score in results:
            writer.writerow([qw, neighbor, score])