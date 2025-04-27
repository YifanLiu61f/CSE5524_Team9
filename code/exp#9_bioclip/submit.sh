module load miniconda3/24.1.2-py310
module load cuda/11.8.0
source activate 5524env

python train_bioclip.py \
  --train-csv ../../data/train/annotations.csv \
  --out       ../../output/exp9_bioclip \
  --batch     32 \
  --epochs    40 \
  --freeze    5 \
  --lr-head   1e-3 \
  --lr-back   3e-5 \
  --workers 4

  python infer_bioclip.py \
      --test-csv  ../../data/test/annotations.csv \
      --checkpoint ../../output/exp9_bioclip/best.pth \
      --output     submission_exp9_bioclip.csv