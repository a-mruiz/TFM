#!/bin/bash
##----------------------- Start job description -----------------------
#SBATCH --partition=standard-gpu
#SBATCH --job-name=TrainNet
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=10
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --mail-user=a.mruiz@upm.es
#SBATCH --mail-type=ALL
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:a100:1
##------------------------ End job description ------------------------

#module purge && module load Python

#srun python3 test_net_gdem_data.py

srun python3 main_train_new.py

#srun python3 ExploreDatasetParameters.py

#srun python3 test_transfer_learning.py

#srun python3 test_quantization.py