#!/usr/bin/env python3

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from transformers import BertTokenizerFast, BertModel

# sys.argv = ["make_similarity_matrix_bert.py", "model_dir"]
MODEL_DIR = sys.argv[1] # pass in the model directory 

tokenizer = BertTokenizerFast.from_pretrained(MODEL_DIR)
model = BertModel.from_pretrained(MODEL_DIR)
model.eval()

# Same style as fastText setup (words can be changed)
sim_dict = {
    "to leave/quit": "λείπων",
    "pursue": "διώκω",
    "loud cry": "βοάν",
    "anger": "ὀργὴ",
    "quick": "ταχύς",
    "patience": "μακροθυμία",
    "to run": "τρέχειν",
    "meat/food": "ἐδητύς",
    "cow": "βοῦς",
    "to free": "ἐλευθεροῦν"
}

# Embedding matrix: [vocab_size, hidden_size]
embedding_matrix = model.embeddings.word_embeddings.weight.detach().cpu().numpy()

def get_word_vector(word):
    """
    Returns a single vector for a word.
    If the tokenizer splits the word into multiple subwords,
    average their embedding vectors.
    """
    pieces = tokenizer.tokenize(word)

    if len(pieces) == 0:
        raise ValueError(f"Tokenizer produced no pieces for word: {word}")

    ids = tokenizer.convert_tokens_to_ids(pieces)

    vecs = []
    for tid in ids:
        if tid == tokenizer.unk_token_id:
            print(f"Warning: {word} contains [UNK] token piece")
        vecs.append(embedding_matrix[tid])

    vec = np.mean(vecs, axis=0)
    norm = np.linalg.norm(vec)

    if norm == 0:
        return vec
    return vec / norm


matrix = []
labels = []

for gloss, greek_word in sim_dict.items():
    try:
        vec = get_word_vector(greek_word)
        matrix.append(vec)
        labels.append(gloss)
        print(f"{greek_word} -> {tokenizer.tokenize(greek_word)}")
    except Exception as e:
        print(f"Skipping {greek_word}: {e}")

matrix = np.array(matrix)
sim_matrix = np.dot(matrix, matrix.T)

plt.figure(figsize=(10, 8))
plt.imshow(sim_matrix, aspect="auto")
plt.colorbar()

plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
plt.yticks(range(len(labels)), labels)

for i in range(len(labels)):
    for j in range(len(labels)):
        plt.text(j, i, f"{sim_matrix[i, j]:.2f}", ha="center", va="center")

plt.title("BERT Static Embedding Similarity Matrix")
plt.xlabel("Words")
plt.ylabel("Words")
plt.tight_layout()

out_path = os.path.join(MODEL_DIR, "bert_similarity_matrix_heatmap.png")
plt.savefig(out_path, dpi=300)
plt.close()

print(f"Saved heatmap to: {out_path}")