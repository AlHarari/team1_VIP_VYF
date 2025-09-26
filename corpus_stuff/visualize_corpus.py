####################################################
## This script compares the word frequencies read ##
## in by the model and how many space-separated   ##
## strings there are.                             ##
####################################################
import fasttext
import numpy as np

def vec2tsv_row(v: np.ndarray) -> str:
    np.set_printoptions(floatmode="fixed", linewidth=500)
    print(str(v))
    return str(v) # Remove last bracket

# Load in latest model.
model_path = "/storage/ice-shared/vip-vyf/embeddings_team/models/array_models_2947993/model_1/model" 
model = fasttext.load_model(model_path)

with open("words_list.tsv", "w", encoding="utf-8") as text_dump_file, open("vectors_list.tsv", "w", encoding="utf-8") as vectors_dump_file:
    words_list = model.get_words()
    for word in words_list[:100]:
        text_dump_file.write(word + "\n")
        # vectors_dump_file.write(vec2tsv_row(model[word]) + "\n")
