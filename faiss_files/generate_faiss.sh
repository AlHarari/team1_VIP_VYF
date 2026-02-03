#!/bin/bash

#SBATCH --job-name=faiss_index
#SBATCH --output=faiss_index.out
#SBATCH --error=faiss_index.err
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mwarren64@gatech.edu

cd /home/hice1/mwarren64/vipteam-folder/team1_VIP_VYF
module load anaconda3
conda init
conda activate embeddings

srun python generate_faiss.py
