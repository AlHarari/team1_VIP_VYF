#!/bin/bash

# Thanks to the answer from https://stackoverflow.com/a/62623593 on how to create a dependency
JOB_ID=$(sbatch --parsable train.sh)
echo $JOB_ID
sbatch --dependency afterok:$JOB_ID --wrap "python after_train.py $JOB_ID"