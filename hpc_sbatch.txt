#!/bin/bash
#BATCH --verbose
#SBATCH --job-name=rfnCMBPTT
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --tasks-per-node=6
#SBATCH --mem=50GB
#SBATCH --gres=gpu:1
#SBATCH --mail-user=nvw224@nyu.edu
module purge

module load pytorch/0.1.12_2 
module load cudnn/8.0v6.0
module load tensorflow/python2.7/20170707
module load pillow/intel/4.0.0
module load scikit-learn/intel/0.18.1 
module load python/intel/2.7.12
module load opencv/intel/3.2
module load keras/2.0.2
module load cuda/8.0.44

python -u classify_movement.py "save_path" "Pitch Type"
