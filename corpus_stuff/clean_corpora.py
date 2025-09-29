import re

dirty_corpus_file = "/storage/ice-shared/vip-vyf/embeddings_team/corpora/corpora.bin"
clean_corpus_file = "/storage/ice-shared/vip-vyf/embeddings_team/corpora/clean_corpora.bin" 

with open(clean_corpus_file, "wb") as output_file_bin:
    with open(dirty_corpus_file, "r", encoding="utf-8") as input_file:

        content = input_file.read()
        print(re.search("[A-Za-z]", content).groups())
        greek_content = re.sub(r"\W?\W?[A-Za-z]\W?\W?", "", content)

        text_r = re.sub(r"\b[^\w\s]", r" \g<0>", greek_content) # punctuations or signs that occur at the end of a word.
        final_greek = re.sub(r"[^\w\s]\b", r"\g<0> ", text_r)        # punctuations or signs that occur at the beginning of a word.

        output_file_bin.write(final_greek.encode("utf-8", errors="ignore"))

with open(dirty_corpus_file, "r", encoding="utf-8") as f:
    content = f.read()
    match_object = re.findall("[A-Za-z]", content)
    print("Number of latin characters in original corpus: ", len(match_object))

with open(clean_corpus_file, "r", encoding="utf-8") as f:
    content = f.read()
    match_object = re.findall("[A-Za-z]", content)
    print("Number of latin characters in new corpus: ", len(match_object))

