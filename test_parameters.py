# Remember that you should be working on the branch
# gensim.

"""
    Here is the relevant documentation: 
    https://radimrehurek.com/gensim/auto_examples/tutorials/run_fasttext.html
"""

from gensim.models.fasttext import FastText
# This is used for setting the "path" to the corpus.
from gensim.test.utils import datapath 

corpus = datapath(
    "/storage/ice-shared/vip-vyf/team1/canonical_greeklit/post_process/" \ 
    "corpora.bin"
    )

new_model = FastText(vector_size=100)
new_model.build_vocab(corpus_file=corpus)
new_model.train(
    corpus_file=corpus,
    epochs=,
    total_words=model.corpus_total_words,
)

