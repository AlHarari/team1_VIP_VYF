import fasttext
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

current_directory = os.getcwd()

model = fasttext.load_model(f"/home/hice1/amohammed87/vipteam-folder/models/model_3")

sample = {'run': 'τρέξιμο', 'think': 'νομίζω', 'hold': 'αμπάρι'}

# Compare similarity w dot product
# We will use 10 similar words to run
# First define and grab our similar words

sim_dict = {'run':'τρέξιμο', 'jog':'σκούντημα', 'sprint':'τρέχω', 
            'walk':'βόλτα', 'jump':'άλμα',
            'love':'αγάπη', 'curse':'κατάρα',
            'hit':'επιτυχία', 'speak':'μιλώ'}

matrix = []

for w in sim_dict.keys():
    word = sim_dict[w] # erronous code had it so that word = sim_dict[w][0]
    vec = model.get_word_vector(word)
    vec = vec / np.linalg.norm(vec) # normalize
    matrix.append(vec)

words = list(sim_dict.keys())

matrix = np.array(matrix)
#print("DOT PRODUCT\n\n", np.dot(matrix, matrix.T))
matrix = np.dot(matrix, matrix.T)

plt.figure(figsize=(10,8))
sns.heatmap(matrix, annot=True, fmt='.2f', cmap='viridis',
            xticklabels=words, yticklabels=words)
plt.title("Similarity matrix heatmap")
plt.xlabel("Words")
plt.ylabel("Words")
plt.xticks(rotation=45, ha='right')

plt.savefig("/home/hice1/amohammed87/vipteam-folder/models/similarity_matrix_heatmap(4).png")

plt.close()


