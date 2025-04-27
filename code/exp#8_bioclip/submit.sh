module load miniconda3/24.1.2-py310
module load cuda/11.8.0
source activate 5524env

python train_bioclip.py \
  --train-csv ../../data/train/annotations.csv \
  --out       ../../output/exp8_bioclip \
  --batch     16 \
  --epochs    40 \
  --freeze    5 \
  --lr-head   1e-3 \
  --lr-back   1e-4