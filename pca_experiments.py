import fasttext
import numpy as np
import matplotlib.pyplot as plt

model_path = "/storage/ice1/0/7/amohammed87/model_9"

model = fasttext.load_model(model_path)

"""
Input matrix:  (2142237, 60)
Output matrix:  (142237, 60)
"""

# Seems like we'll be using the ouptut matrix.
# input_vectors = model.get_input_matrix() # Returns a numpy array.
# print("Input matrix: ", input_vectors.shape)

output_vectors = model.get_output_matrix()
print("Output matrix: ", output_vectors.shape)

# Apply PCA to project everything onto subspace spanned by first 2 principal component vectors.
U, S, V_t = np.linalg.svd(output_vectors, full_matrices=False)

