module load miniconda3/24.1.2-py310
module load cuda/11.8.0
source activate 5524env

python ensemble_infer.py \
       --test-csv  ../../data/test/annotations.csv \
       --dino-ckpt ../../output/exp7_dinov2/epoch_23.pth \
       --bio-ckpt  ../../output/exp9_bioclip/best.pth \
       --out       ensemble_submission.csv \
       --batch     64


python ensemble_infer.py \
       --test-csv  ../../data/test/annotations.csv \
       --dino-ckpt ../../output/exp7_dinov2/epoch_23.pth \
       --bio-ckpt  ../../output/exp10_bioclip/best.pth \
       --out       ensemble_2_submission.csv \
       --batch     64