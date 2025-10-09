#################################################################
## This script spits out files that are used for visualization ##
## purposes and also for database generation purposes.         ##
#################################################################
import fasttext
import numpy as np

common_path = "/storage/ice-shared/vip-vyf/embeddings_team/models/array_models_3181969/model_9/"
output_path = common_path
model_path = common_path + "model"
model = fasttext.load_model(model_path)
DIM = model.get_dimension() 
input_matrix = np.zeros((1, DIM)) 

with open(output_path + "words_list.txt", "w", encoding="utf-8") as text_dump_file, 
    # open(output_path + f"vectors_list_{DIM}.txt", "w", encoding="utf-8") as vectors_dump_file_txt,
    open(output_path + f"vectors_list_{DIM}.tsv", "w", encoding="utf-8") as vectors_dump_file_tsv:
    words_list = model.get_words()
    for word in words_list:
        text_dump_file.write(word + "\n")
        vector = model.get_word_vector(word)
        vector_tsv = '\t'.join(str(x) for x in vector)
        vectors_dump_file_tsv.write(vector_tsv)
        vectors_dump_file_tsv.write("\n")        # vectors_dump_file.write(vec2tsv_row(model[word]) + "\n")
        # vectors_dump_file_txt.write("[" + ", ".join(str(x) for x in vector) + "]")
        # vectors_dump_file_txt.write("\n")
        input_matrix = np.concatenate((input_matrix, vector.reshape((1, DIM))))

input_matrix = input_matrix[1:,:] # Remove 0 row
np.save(output_path + f"input_matrix_{DIM}.npy", input_matrix)
