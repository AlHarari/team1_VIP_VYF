import os
import pickle as pkl

import fasttext
import numpy as np


greek_model_path = "/storage/ice-shared/vip-vyf/embeddings_team/models/array_models_3181969/model_9/model"
english_vec_path = "/storage/ice-shared/vip-vyf/embeddings_team/wiki-news-300d-1M-subword.vec"
seed_dict_path = "/home/hice1/mwarren64/vip-folder/team1_VIP_VYF/BLI_files/dictionary.pkl"
output_path = "W_greek_to_english.npy"


def load_english_vectors(vec_path):
    english_words = []
    english_vectors = []

    with open(vec_path, "r", encoding="utf-8", errors="ignore") as file:
        first_line = file.readline().strip().split()

        if len(first_line) == 2 and all(part.isdigit() for part in first_line):
            vector_dim = int(first_line[1])
        else:
            word = first_line[0]
            values = np.asarray(first_line[1:], dtype=np.float32)
            vector_dim = values.shape[0]

            english_words.append(word)
            english_vectors.append(values)

        for line in file:
            parts = line.rstrip().split(" ")
            if len(parts) < vector_dim + 1:
                continue

            word = parts[0]
            values = np.asarray(parts[1:], dtype=np.float32)

            english_words.append(word)
            english_vectors.append(values)

    english_vectors = np.vstack(english_vectors)
    return english_vectors, english_words


def normalize_rows(matrix):
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = norms + 1e-12
    return matrix / norms


def main():
    greek_model = fasttext.load_model(greek_model_path)
    english_vectors, english_words = load_english_vectors(english_vec_path)
    english_vectors = normalize_rows(english_vectors)
    english_word_to_index = {}
    
    for index, word in enumerate(english_words):
        english_word_to_index[word] = index

    with open(seed_dict_path, "rb") as file:
        seed_dictionary = pkl.load(file)

    greek_vocabulary = set(greek_model.get_words())

    greek_training_vectors = []
    english_training_vectors = []
    num_pairs = 0

    for greek_word, english_set in seed_dictionary.items():
        if greek_word not in greek_vocabulary:
            continue

        greek_vector = greek_model.get_word_vector(greek_word)

        for english_word in english_set:
            if english_word not in english_word_to_index:
                continue

            english_index = english_word_to_index[english_word]
            english_vector = english_vectors[english_index]

            greek_training_vectors.append(greek_vector)
            english_training_vectors.append(english_vector)
            num_pairs += 1

    X = np.asarray(greek_training_vectors, dtype=np.float32)
    Y = np.asarray(english_training_vectors, dtype=np.float32)

    X = normalize_rows(X)
    Y = normalize_rows(Y)

    cross_covariance = X.T @ Y
    U, _, Vt = np.linalg.svd(cross_covariance)
    W = U @ Vt

    np.save(output_path, W)

main()

