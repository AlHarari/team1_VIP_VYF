####################################################
## This script compares the word frequencies read ##
## in by the model and how many space-separated   ##
## strings there are.                             ##
####################################################
import fasttext
from matplotlib import pyplot as plt

# Load in latest model.
model_path = "/home/hice1/amohammed87/scratch/model_9" 
model = fasttext.load_model(model_path)
print(dir(model))
