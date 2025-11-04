"""
    This file is meant to build the dictionary, which we will use later to perform BLI.
    We've used the IAA-all-TPs.csv file from the following repository:
    https://github.com/UgaritAlignment/Alignment-Gold-Standards/
"""
from collections import defaultdict
import pickle as pkl

dictionary = defaultdict(set)
with open("IAA-all-TPs.csv", "r") as tps_file:
    non_header_lines = tps_file.readlines()[2:] 
    for line in non_header_lines:
        row = line.split(",")
        greek_word_chiara, translation_chiara, greek_word_farnoosh, translation_farnoosh = row[1], row[2].strip(), row[5], row[6].strip()
        if greek_word_chiara != "":
            dictionary[greek_word_chiara].add(translation_chiara) 
        if greek_word_farnoosh != "":
            dictionary[greek_word_farnoosh].add(translation_farnoosh)

with open("dictionary.pkl", "wb") as dictionary_file:
    pkl.dump(dictionary, dictionary_file)