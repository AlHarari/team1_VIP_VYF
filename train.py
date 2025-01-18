import sys
import os
from contextlib import redirect_stdout, redirect_stderr
from fasttext import train_unsupervised

current_directory = os.getcwd()
model_path = os.path.join(current_directory, "models")
log_path = '/home/hice1/dharden7/scratch/canonical_greeklit/training_log.txt'

def train_model(model_path, log_path):
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    with open(log_path, 'w') as f:
        with redirect_stdout(f), redirect_stderr(f):
            print("TRY SUCCESS")
            model = train_unsupervised(
                input='/home/hice1/dharden7/scratch/canonical_greeklit/post_process/corpus_utf8.bin',
                model='skipgram',
                lr=0.05,
                dim=200,
                ws=10,
                epoch=40,
                minCount=5,
                minn=3,
                maxn=10,
                neg=8,
                wordNgrams=2,
                loss='ns',
                bucket=2000000,
                thread=6,
                lrUpdateRate=100,
                t=0.0001,
                verbose=2
            )
    return model

v_num = 2
model = train_model(model_path, log_path)
if model is not None:
    model.save_model(f"{model_path}/model_{v_num}")
else:
    print("\nMODEL TRAINING FAILED.")

