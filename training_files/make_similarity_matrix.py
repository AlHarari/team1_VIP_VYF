import fasttext
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Try on array_models_2947993, model 1
# and array_models_3181969, model 0

# sys.argv = ["make_similarity_matrix.py", "task id", "job id"]
TASK_ID, JOB_ID = sys.argv[1], sys.argv[2]
PATH_TO_DIR = f"/storage/ice-shared/vip-vyf/embeddings_team/models/array_models_{JOB_ID}/model_{TASK_ID}"
model = fasttext.load_model(PATH_TO_DIR + "/model")

# Compare similarity w dot product
# We will use 10 similar words to run
# First define and grab our similar words

# sim_dict = {'run':'τρέξιμο', 'jog':'σκούντημα', 'sprint':'τρέχω', 
#             'walk':'βόλτα', 'jump':'άλμα',
#             'love':'αγάπη', 'curse':'κατάρα',
#             'hit':'επιτυχία', 'speak':'μιλώ'}

sim_dict = {
    "to leave/quit": "λείπων",
    "pursue": "διώκω",
    "loud cry": "βοάν",
    "anger": "ὀργὴ",
    "quick": "ταχύς",
    "patience": "μακροθυμία",
    "to run" : "τρέχειν", # Infinitive form. Not root word.
    "meat/food": "ἐδητύς",
    "cow": "βοῦς",
    "to free": "ἐλευθεροῦν"
}

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

plt.savefig(PATH_TO_DIR + "/similarity_matrix_heatmap_in_words.png") # Thinking about adding more than one similarity matrix per model.

plt.close()


