import sys
import os
from fasttext import train_unsupervised
from helper_file import get_arguments 

# sys.argv = ["train.py", "task id", "array job id"]
TASK_ID, JOB_ID = sys.argv[1], sys.argv[2]
model_path = f"/storage/ice-shared/vip-vyf/embeddings_team/models/array_models_{JOB_ID}/model_{TASK_ID}"

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

LR, DIM, WS, EPOCH, MINC, MINN, MAXN, NEG, WORDN = get_arguments(sys.argv[:2])
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

if model is not None:
    model.save_model(f"{model_path}/model")
else:
    print("\nMODEL TRAINING FAILED.")

