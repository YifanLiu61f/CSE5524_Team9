module load miniconda3/24.1.2-py310
module load cuda/11.8.0
source activate 5524env

python train_bioclip.py \
  --train-csv ../../data/train/annotations.csv \
  --out       ../../output/exp10_bioclip \
  --batch     64 \
  --epochs    60 \
  --freeze    6  \
  --lr-head   5e-4 \
  --lr-back   5e-5 \
  --wd        5e-3 \
  --grad-accum 2 \
  --workers   8

  python infer_bioclip.py \
      --test-csv  ../../data/test/annotations.csv \
      --checkpoint ../../output/exp10_bioclip/best.pth \
      --output     submission_exp10_bioclip.csv