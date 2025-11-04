#!/bin/bash
#SBATCH --job-name=get_centroid_neighbore
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH -o /home/hice1/mwarren64/vip-folder/logs/get_centroid_words_%A_%a.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mwarren64@gatech.edu



cd /home/hice1/mwarren64/vip-folder/team1_VIP_VYF

module load anaconda3
conda init
source ~/.bashrc
conda activate embeddings

srun python centroids_as_words.py

