import fasttext
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

model = fasttext.load_model(f"~/scratch/model_9")

# Compare similarity w dot product
# We will use 10 similar words to run
# First define and grab our similar words

sim_dict = {'run':'τρέξιμο', 'jog':'σκούντημα', 'sprint':'τρέχω', 
            'walk':'βόλτα', 'jump':'άλμα',
            'love':'αγάπη', 'curse':'κατάρα',
            'hit':'επιτυχία', 'speak':'μιλώ'}

matrix = []

for w in sim_dict.keys():
    word = sim_dict[w] 
    vec = model.get_word_vector(word)
    vec = vec / np.linalg.norm(vec) # normalize
    matrix.append(vec)

words = list(sim_dict.keys())

matrix = np.array(matrix)
matrix = np.dot(matrix, matrix.T)

plt.figure(figsize=(10,8))
sns.heatmap(matrix, annot=True, fmt='.2f', cmap='viridis',
            xticklabels=words, yticklabels=words)
plt.title("Similarity matrix heatmap")
plt.xlabel("Words")
plt.ylabel("Words")
plt.xticks(rotation=45, ha='right')

plt.savefig("../visualizations/similarity_matrix_heatmap_slurm.png")

plt.close()


