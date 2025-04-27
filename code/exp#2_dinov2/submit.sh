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
python train_dinov2.py \
  --train-csv ../../data/train/annotations.csv \
  --output-dir ../../output/exp2_dinov2 \
  --epochs 20 \
  --batch-size 16

# infer
  python infer.py \
  --test-csv ../../data/test/annotations.csv \
  --checkpoint ../../output/exp2_dinov2/epoch_20.pth \
  --output submission_exp2_epoch20.csv \
  --batch-size 64 \
  --model-id facebook/dinov2-base

    python infer.py \
  --test-csv ../../data/test/annotations.csv \
  --checkpoint ../../output/exp2_dinov2/epoch_19.pth \
  --output submission_exp2_epoch19.csv \
  --batch-size 64 \
  --model-id facebook/dinov2-base

      python infer.py \
  --test-csv ../../data/test/annotations.csv \
  --checkpoint ../../output/exp2_dinov2/epoch_18.pth \
  --output submission_exp2_epoch18.csv \
  --batch-size 64 \
  --model-id facebook/dinov2-base

      python infer.py \
  --test-csv ../../data/test/annotations.csv \
  --checkpoint ../../output/exp2_dinov2/epoch_15.pth \
  --output submission_exp2_epoch15.csv \
  --batch-size 64 \
  --model-id facebook/dinov2-base

    python infer.py \
  --test-csv ../../data/test/annotations.csv \
  --checkpoint ../../output/exp4_40epoch_dinov2/epoch_35.pth \
  --output ../exp#4_dinov2_40epoch/submission_exp4_epoch35.csv \
  --batch-size 64 \
  --model-id facebook/dinov2-base
# did not run the bash script but the separate python commands work in terminal