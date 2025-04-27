#!/bin/bash
#SBATCH --job-name=5524
#SBATCH --account=PAS2985            
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1                 # ask for one A100
#SBATCH --cpus-per-task=8                 # for DataLoader workers
#SBATCH --mem=50G                         # RAM
#SBATCH --time=05:00:00                   # hh:mm:ss
#SBATCH --mail-type=ALL
#SBATCH --mail-user=liu.11275@osu.edu
#SBATCH --output=logs/5524_%j.out
#SBATCH --error=logs/5524_%j.err

# Load modules (must specify versions on Ascend)
module load miniconda3/24.1.2-py310
module load cuda/11.8.0

# Activate your virtualenv or conda
source activate 5524env

# Move into your project directory
cd $SLURM_SUBMIT_DIR

# 1) TRAIN on the full dataset
python train.py   --train-csv ../../data/train/annotations.csv   --output-dir ../../output/exp3_dinov2   --epochs 40 --patience 5 --workers 4 --batch-size 16 --save-every 5

# infer


# did not run the bash script but the separate python commands work in terminal