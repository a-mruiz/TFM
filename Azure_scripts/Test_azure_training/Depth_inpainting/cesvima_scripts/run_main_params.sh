#!/bin/bash
##----------------------- Start job description -----------------------
#SBATCH --partition=standard-gpu
#SBATCH --job-name=HParams
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=14
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --mail-user=a.mruiz@upm.es
#SBATCH --mail-type=ALL
#SBATCH --time=64:00:00
#SBATCH --gres=gpu:a100:1
##------------------------ End job description ------------------------

#module purge && module load Python

#srun python3 test_net_gdem_data.py

srun python3 main_train_nets_cesvima_hyperparameter_tunning.py

#srun python3 ExploreDatasetParameters.py

#srun python3 test_transfer_learning.py

#srun python3 test_quantization.py