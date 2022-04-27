#!/bin/bash
##----------------------- Start job description -----------------------
#SBATCH --partition=standard-gpu
#SBATCH --job-name=TrainNet
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=5
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --mail-user=a.mruiz@upm.es
#SBATCH --mail-type=ALL
#SBATCH --time=0:05:00
#SBATCH --gres=gpu:v100:1
##------------------------ End job description ------------------------

#module purge && module load Python

#srun python3 test_net_gdem_data.py

srun python3 main_test_net_gdem_data.py

#srun python3 ExploreDatasetParameters.py

#srun python3 test_transfer_learning.py

#srun python3 test_quantization.py