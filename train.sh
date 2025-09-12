#!/bin/bash

#SBATCH -JTrainingModels
#SBATCH --ntasks 1 --cpus-per-task 24 ## Each job is made of one task, which exists on 1 node and uses 24 cpus. Even though each task lives on one node, it doesn't mean that one node only works on 1 task. I sort of want it to be that way, however
#SBATCH --mem-per-cpu=4G
#SBATCH -t720 ## Format has to be D-HH:MM:SS, or just how many minutes. I know, 12 hours is too much.
#SBATCH -o ../Reports/Report-%A_%a.out # Let's test this out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=amohammed87@gatech.edu
#SBATCH --array=0-9

## Just to be sure.
cd /home/hice1/amohammed87/vipteam-folder/team1_VIP_VYF

module load anaconda3
conda init
source ~/.bashrc         ## Don't know if I need to include this anymore.
conda activate team1-vip

srun python train.py $SLURM_ARRAY_TASK_ID $SLURM_ARRAY_JOB_ID > ../training_logs/slurm${SLURM_ARRAY_TASK_ID}.txt 2>&1 && python make_similarity_matrix.py $SLURM_ARRAY_TASK_ID $SLURM_ARRAY_JOB_ID && cp ./input_arguments.txt /storage/ice-shared/vip-vyf/embeddings_team/models/array_models_${SLURM_ARRAY_JOB_ID}/ 
