import os 
import numpy as np
import fasttext

# Load the similarity matrix
similarity_matrix = np.load('similarity_matrix.npy')

# Load the word list
with open('word_list.txt', 'r', encoding='utf-8') as f:
    selected_words = [line.strip() for line in f]

# Recreate the word-to-index mapping
word_to_index = {word: idx for idx, word in enumerate(selected_words)}

def get_similarity(word1, word2):
    idx1 = word_to_index.get(word1)
    idx2 = word_to_index.get(word2)
    if idx1 is None or idx2 is None:
        return None  # Or handle unknown words appropriately
    return similarity_matrix[idx1, idx2]

word1 = 'τρέξιμο'  # Greek word for 'run'
word2 = 'περπατώ'  # Greek word for 'walk'
print('Before')
similarity = get_similarity(word1, word2)
print('After')
print(f"The similarity between '{word1}' and '{word2}' is: {similarity}")




