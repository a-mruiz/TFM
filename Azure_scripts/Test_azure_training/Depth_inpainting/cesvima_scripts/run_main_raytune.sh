#!/bin/bash
##----------------------- Start job description -----------------------
#SBATCH --partition=standard-gpu
#SBATCH --job-name=HParams
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --mem=40G
#SBATCH --mail-user=a.mruiz@upm.es
#SBATCH --mail-type=ALL
#SBATCH --time=128:00:00
#SBATCH --gres=gpu:a100:2
##------------------------ End job description ------------------------

#module purge && module load Python

srun python3 main_train_new_raytune.py
