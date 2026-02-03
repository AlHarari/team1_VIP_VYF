#!/bin/bash
#SBATCH --job-name=spectral_clustering_neighbors
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH -o /home/hice1/mwarren64/vip-folder/logs/cluster_2__%A_%a.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mwarren64@gatech.edu



cd /home/hice1/mwarren64/vip-folder/team1_VIP_VYF/BLI_files

module load anaconda3
conda init
source ~/.bashrc
conda activate embeddings

srun python BLI_translator.py

