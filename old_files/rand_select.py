import fasttext
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

current_directory = os.getcwd()

model_path = "/storage/ice1/0/7/amohammed87/" # os.path.join(current_directory, "models")

model = fasttext.load_model(f"{model_path}/model_9")

num_threads = 4

# os.environ['OPENBLAS_NUM_THREADS'] = 'num_threads'  # Replace '8' with the number of your CPU cores
# os.environ['MKL_NUM_THREADS'] = 'num_threads'
# os.environ['NUMEXPR_NUM_THREADS'] = 'num_threads'
# os.environ['OMP_NUM_THREADS'] = 'num_threads'

all_words = model.get_words()

# Select top 10,000 words
random_selection = all_words[:10000]

import random

# Select 10,000 random words
# random_selection = random.sample(all_words, 10000)

word_to_index = {word: idx for idx, word in enumerate(random_selection)}

vectors = np.array([model.get_word_vector(word) for word in random_selection])

# Compute similarity matrix 
similarity_matrix = np.dot(vectors, vectors.T)

# Save the similarity matrix
np.save('similarity_matrix.npy', similarity_matrix)

# Save the word list
with open('word_list.txt', 'w') as f:
    for word in random_selection:
        f.write(word + '\n')


