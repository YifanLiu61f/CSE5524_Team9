# Team 9 Final Project — CVPR-FGVC (FathomNet 2025)

**Competition**  
[FathomNet 2025 @ CVPR-FGVC (Kaggle)](https://www.kaggle.com/competitions/fathomnet-2025/overview)


## Leaderboard Snapshot

![Our Entry on the Kaggle Leaderboard](image.png)

## Installation

1. **Clone this repository**  
   ```bash
   git clone https://github.com/YifanLiu61f/fgvc-comp-2025.git
   cd fathomnet-2025-team9

2.	**Create & activate your Python environment**
We recommend Python 3.8+ and conda:
    ```bash
    conda create -n fmn2025 python=3.8 -y
    conda activate fmn2025

3.	**Install dependencies**
    ```bash
    pip install torch torchvision transformers pillow tqdm

## Usage

1.	**Prepare test CSV**

    data/test/annotations.csv should list your test ROIs:

    path,label
    data/test/rois/1_1.png,
    data/test/rois/2_2.png,
    data/test/rois/2_3.png,
    ...

2.	**Run inference**
    From the project root:
    ```bash
    python infer.py \
    --test-csv   data/test/annotations.csv \
    --checkpoint epoch_23.pth \
    --output     submission.csv \
    --batch-size 64 \
    --model-id   facebook/dinov2-base

**This will write your predictions to submission.csv in the format:**
   
    annotation_id,concept_name
    1,Pennatula phosphorea
    2,Funiculina
    3,Funiculina
    4,Funiculina
    ...


## Advanced Algorithm

We fine-tune the self-supervised DINOv2 ViT backbone (facebook/dinov2-base) on FathomNet ROIs:

	Warm-up: train only the newly-added linear head for 3 epochs (LR=1e-3).

	Full fine-tuning: unfreeze the backbone, train backbone @1e-5 with cosine-annealed AdamW + label smoothing (ε=0.1), mixed-precision.

	Checkpoint: our best model is in epoch_23.pth (30 epochs in total, 89.1% val top-1, 2.73 test score).

## Test Examples

We ship a small subset of validation ROIs under data/test/rois/. Feel free to add more images & update annotations.csv accordingly.

## Writing Your Own Inference

The core inference logic lives in infer.py, which:
	1.	Loads label_map.json (saved alongside epoch_23.pth).
	2.	Applies the DINOv2 preprocessing pipeline (AutoImageProcessor + torchvision.transforms).
	3.	Runs the model and writes a Kaggle-format CSV.

## License

This project is released under the MIT License.
© 2025 CSE 5524 Team 9, The Ohio State University.
