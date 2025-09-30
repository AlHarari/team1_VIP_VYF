######################################################
## This script produces the words_list.tsv file and ##
## the vec_list.tsv file.                           ##
######################################################
import fasttext
import numpy as np

# Load in latest model.
model_path = "/storage/ice-shared/vip-vyf/embeddings_team/models/array_models_3181969/model_0/model" 
model = fasttext.load_model(model_path)

with open("words_list.tsv", "w", encoding="utf-8") as text_dump_file, open("vectors_list.tsv", "w", encoding="utf-8") as vectors_dump_file:
    words_list = model.get_words()
    for word in words_list:
        text_dump_file.write(word + "\n")
        #np.savetxt("vectors_list.tsv", model[word].T, delimiter="\t")
