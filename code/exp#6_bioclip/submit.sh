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
python train.py \
  --model-backend openclip \
  --model-id hf-hub:imageomics/bioclip \
  --train-csv ../../data/train/annotations.csv \
  --output-dir ../../output/exp6_bioclip \
  --batch-size 16 \
  --freeze-epochs 8 \
  --lr-head 1e-3 \
  --lr-backbone 3e-4 \
  --epochs 40  --patience 8  --save-every 5 \
  --workers 4
  
# infer
python infer_bioclip.py \
      --test-csv  ../../data/test/annotations.csv \
      --checkpoint ../../output/exp6_bioclip/best.pth \
      --output     submission_bioclip_bestpth.csv

# did not run the bash script but the separate python commands work in terminal