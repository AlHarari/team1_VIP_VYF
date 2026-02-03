import fasttext
import faiss
import numpy as np

# This model performed the best so far
path_to_model = "/storage/ice-shared/vip-vyf/embeddings_team/models/array_models_2947993/model_1/model" 
model = fasttext.load_model(path_to_model)
input_matrix = model.get_input_matrix()
words = model.get_words()

dim = model.get_dimension()
input_matrix = model[words[0]].reshape((1, dim))
i = 0
for word in words[1:]:
    print(i)
    input_matrix = np.concatenate((input_matrix, model[word].reshape((1, dim))))
    i += 1

rows, cols = input_matrix.shape
query_matrix = input_matrix[1, :].reshape((1, cols)) # kai

# Build the index.
idx = faiss.IndexFlatIP(cols)
idx.add(input_matrix)

# Specify how many neighbors.
k = 5
_, index_matrix = idx.search(query_matrix, k)

print("WORD CHOSEN: ")
for row in index_matrix: # Making an assumption here: word[i]'s vector embedding is the ith row of input_matrix.
    print(row) # Sometimes, we get a subqord that has a high similarity, but this means that it's index is beyond words' size.
    for idx in row:
        print(words[idx])

print("Native implementation")
print(model.get_nearest_neighbors(words[1], k=5))
