#!/bin/bash
#SBATCH -J fasttext                # job name
#SBATCH -N1 --ntasks-per-node=6       # number of nodes and cores per node required
#SBATCH --mem-per-cpu=8G              # memory per core
#SBATCH -t1                           # duration of the job (ex: 5 min)
#SBATCH -oReport-%j.out               # output file name

cd /home/hice1/dharden7/scratch/canonical_greeklit

module load anaconda3

# conda init

# Initialize Conda
source /usr/local/pace-apps/manual/packages/anaconda3/2023.03/etc/profile.d/conda.sh

# Activate your specific environment
conda activate /home/hice1/dharden7/.conda/envs/parse

export OMP_NUM_THREADS=6
export MKL_NUM_THREADS=6
export NUMEXPR_NUM_THREADS=6
export OPENBLAS_NUM_THREADS=6

python t1.py > training_log.txt 2>&1

