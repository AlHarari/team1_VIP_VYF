import fasttext
import numpy as np
import pickle as pkl

greek_model_path = "/storage/ice-shared/vip-vyf/embeddings_team/models/array_models_3181969/model_9/model"
english_vec_path = "/storage/ice-shared/vip-vyf/embeddings_team/wiki-news-300d-1M-subword.vec"
mapping_path = "/home/hice1/mwarren64/vip-folder/team1_VIP_VYF/BLI_files/W_greek_to_english.npy"

greek_model = fasttext.load_model(greek_model_path)
W = np.load(mapping_path)

def load_english_vectors(vec_path):
    words = []
    vecs = []
    with open(vec_path, "r", encoding="utf-8", errors="ignore") as f:
        first = f.readline().split()
        if len(first) == 2 and all(x.isdigit() for x in first):
            dim = int(first[1])
        else:
            words.append(first[0])
            vecs.append(np.array(first[1:], float))
            dim = len(vecs[-1])
        for line in f:
            parts = line.rstrip().split(" ")
            if len(parts) < dim + 1:
                continue
            words.append(parts[0])
            vecs.append(np.array(parts[1:], float))
        vecs = np.vstack(vecs)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
        return np.array(vecs, np.float32), words

english_vecs, english_words = load_english_vectors(english_vec_path)

def translate(word, top_k=5):
    greek_vec = greek_model.get_word_vector(word).astype(np.float32)
    greek_vec /= np.linalg.norm(greek_vec) + 1e-12
    mapped_vec = greek_vec @ W
    mapped_vec /= np.linalg.norm(mapped_vec) + 1e-12
    scores = english_vecs @ mapped_vec
    best = scores.argsort()[-top_k:][::-1]
    return [(english_words[i], float(scores[i])) for i in best]

greek_words = ["λόγος", "θεός", "ἀνήρ"]
output_path = "possible_greek_translations.txt"
with open(output_path, "w", encoding="utf-8") as f:
    for w in greek_words:
        results = translate(w, top_k=5)
        f.write(f"\n{w}:")
        for eng, score in results:
            f.write(f"  {eng:20s} {score:.4f}\n")
        f.write("\n")
