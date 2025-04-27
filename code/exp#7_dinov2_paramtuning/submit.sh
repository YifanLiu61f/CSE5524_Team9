module load miniconda3/24.1.2-py310
module load cuda/11.8.0
source activate 5524env

python train_dinov2.py \
  --train-csv ../../data/train/annotations.csv \
  --output-dir ../../output/exp7_dinov2 \
  --batch-size 16 \
  --epochs 30 \
  --lr-head 1e-3 \
  --lr-backbone 1e-5 \
  --workers 4

  # infer
  python infer.py \
  --test-csv ../../data/test/annotations.csv \
  --checkpoint ../../output/exp7_dinov2/epoch_23.pth \
  --output ../exp#7_dinov2_paramtuning/submission_exp7_epoch23_30.csv \
  --batch-size 64 \
  --model-id facebook/dinov2-base

  python infer.py \
  --test-csv ../../data/test/annotations.csv \
  --checkpoint ../../output/exp7_dinov2/epoch_26.pth \
  --output ../exp#7_dinov2_paramtuning/submission_exp7_epoch26_30.csv \
  --batch-size 64 \
  --model-id facebook/dinov2-base