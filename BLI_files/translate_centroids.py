"""
    This file uses the dictionary created to translate the words in centroids_neighbors.txt.
    Given a row n_1,n_2,...,n_k\n in the text file, we add a row to a new file, where the new row
    is t(n_1),t(n_2),...,t(n_k)\n, where t(n_i) is either the english translation of the greek word or
    N/A if translation not found.
"""
import pickle as pkl

# Get our dictionary back!
with open("dictionary.pkl", "rb") as pickled_dictionary:
    dictionary = pkl.load(pickled_dictionary)

# Translate! I had to clean centroids_neighbors.txt file, but hopefully using a model trained on the
# new corpus would prevent issues coming up in the first place.
with open("centroids_neighbors.txt", encoding="utf-8") as centroid_neighbors_file, open("translated_centroids_new.csv", "w", encoding="utf-8") as translated_centroid_neighbors_file:
    for line in centroid_neighbors_file.readlines():
        neighbors = line.split(",")
        neighbors_translation = []
        for neighbor in neighbors:
            if neighbor in dictionary:
                neighbors_translation.append(str(dictionary[neighbor]))
            else:
                neighbors_translation.append("N/A")
        translated_centroid_neighbors_file.write("^".join(neighbors_translation) + "\n")
