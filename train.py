import sys
import os
from fasttext import train_unsupervised
from helper_file import get_arguments 

current_directory = os.getcwd()
model_path = "/storage/ice1/0/7/amohammed87/"

"""
    These are the parameters that were used that lead to the most optimal loss 
    value (about 0.098). 
    dim 200
    ws 10
    epoch 350
    minCount 5
    neg 8
    wordNgrams 5
    loss ns
    model sg
    bucket 2000000
    minn 3
    maxn 10
    lrUpdateRate 100
    t 0.0001

    Strangely enough, I was not able to achieve these same results
    using the same corpus. Strange!

    By the way, you can use fasttext dump MODEL args to get this list!
    Reference: https://github.com/facebookresearch/fastText/issues/418#issuecomment-366302469.
"""


# Ok, now start training.
if not os.path.exists(model_path):
    os.makedirs(model_path)

LR, DIM, WS, EPOCH, MINC, MINN, MAXN, NEG, WORDN = get_arguments(sys.argv)

"""
This is the order of the arguments, by the way.
model = train_unsupervised(
    input='/storage/ice-shared/vip-vyf/team1/canonical_greeklit/post_process/corpora.bin',
    model='skipgram',
    lr=0.05,
    dim=200,
    ws=8,
    epoch=1,
    minCount=5,
    minn=3,
    maxn=10,
    neg=8,
    wordNgrams=2,
    loss='ns',
    bucket=2000000,
    thread=64,
    lrUpdateRate=100,
    t=0.0001,
    verbose=2
)
"""

model = train_unsupervised(
    input='/storage/ice-shared/vip-vyf/team1/canonical_greeklit/post_process/corpora.bin',
    model='skipgram',
    lr=LR,
    dim=DIM,
    ws=WS,
    epoch=EPOCH,
    minCount=MINC,
    minn=MINN,
    maxn=MAXN,
    neg=NEG,
    wordNgrams=WORDN,
    loss='ns',
    bucket=2000000,
    thread=24,
    lrUpdateRate=100,
    t=0.0001,
    verbose=2
)

# Fine if model is overriden during job array execution.
v_num = 9
if model is not None:
    model.save_model(f"{model_path}/model_{v_num}")
else:
    print("\nMODEL TRAINING FAILED.")

