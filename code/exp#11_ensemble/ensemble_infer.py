"""
Predict with Dinov2 + BioCLIP and average class-probabilities
-------------------------------------------------------------
$ python ensemble_infer.py \
      --test-csv  ../../data/test.csv \
      --dino-ckpt ../../output/exp7_dinov2/best.pth \
      --bio-ckpt  ../../output/exp10_bioclip/best.pth \
      --out       ensemble_submission.csv
"""

import argparse, csv, json, os, torch
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import AutoImageProcessor, AutoModelForImageClassification
import open_clip
from dataset import CsvFathomDataset
from tqdm.auto import tqdm
import warnings

# ---------- CLI ----------
def get_args():
    pa = argparse.ArgumentParser()
    pa.add_argument("--test-csv",  required=True)
    pa.add_argument("--dino-ckpt", required=True)
    pa.add_argument("--bio-ckpt",  required=True)
    pa.add_argument("--out",       default="ensemble_submission.csv")
    pa.add_argument("--batch",     type=int, default=64)
    pa.add_argument("--dino-id",   default="facebook/dinov2-base")
    return pa.parse_args()

# ---------- misc ----------
def dev():
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")

# ---------- models ----------
def load_dino(model_id, ckpt_path, n_cls, device):
    proc  = AutoImageProcessor.from_pretrained(model_id)
    dino  = AutoModelForImageClassification.from_pretrained(model_id,
                                                            num_labels=n_cls)
    dino.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    dino.to(device).eval()
    tf = transforms.Compose([
        transforms.Resize(proc.size["shortest_edge"]),
        transforms.CenterCrop((proc.crop_size["height"], proc.crop_size["width"])),
        transforms.ToTensor(),
        transforms.Normalize(proc.image_mean, proc.image_std),
    ])
    return dino, tf

def load_bioclip(ckpt_path, n_cls, device):
    backbone, _, tf_val = open_clip.create_model_and_transforms(
        "hf-hub:imageomics/bioclip")
    img_dim = backbone.visual.output_dim

    class BioCLIPClassifier(torch.nn.Module):
        def __init__(self, clip_backbone, n_cls):
            super().__init__()
            self.backbone = clip_backbone
            self.head     = torch.nn.Linear(img_dim, n_cls)
        def forward(self, x):
            with torch.no_grad():
                feats = self.backbone.encode_image(x)
            return self.head(feats)

    model = BioCLIPClassifier(backbone, n_cls)
    state = torch.load(ckpt_path, map_location="cpu")
    # training script saved dict under ['model']
    model.load_state_dict(state["model"] if "model" in state else state, strict=True)
    model.to(device).eval()
    return model, tf_val

# ---------- main ----------
def main():
    args = get_args()
    dv   = dev()

    # ---------- label-map (must match!) ----------
    with open(Path(args.dino_ckpt).parent / "label_map.json") as f:
        cat2idx_d = json.load(f)
    with open(Path(args.bio_ckpt).parent  / "label_map.json") as f:
        cat2idx_b = json.load(f)

    if cat2idx_d != cat2idx_b:
        raise ValueError("label_map.json from the two runs do not match!")
    cat2idx = cat2idx_d
    idx2cat = {v:k for k,v in cat2idx.items()}
    n_cls   = len(cat2idx)

    # ---------- load both models ----------
    dino, tf_dino = load_dino(args.dino_id, args.dino_ckpt, n_cls, dv)
    bio , tf_bio  = load_bioclip(args.bio_ckpt, n_cls, dv)

    # ---------- dataset ----------
    # return raw PIL image path; we’ll apply *both* transforms on-the-fly
    ds = CsvFathomDataset(args.test_csv, transforms=None)  # no transform yet
    def collate(batch):
        pil_imgs, ann_ids = [], []
        for item in batch:
            # item can be (img, path) or (img,_,path) depending on your CsvFathomDataset
            if len(item) == 2: img, aid = item
            else:              img, _, aid = item
            pil_imgs.append(img)
            ann_ids.append(aid)
        return pil_imgs, ann_ids

    loader = DataLoader(ds, batch_size=args.batch, shuffle=False,
                        num_workers=2, collate_fn=collate)

    softmax = torch.nn.Softmax(dim=1)

    # ---------- inference ----------
    rows = []
    with torch.no_grad():
        for pil_imgs, ann_ids in tqdm(loader, desc="Ensemble"):
            # create two stacks of tensors with *their* transforms
            imgs_d = torch.stack([tf_dino(img) for img in pil_imgs]).to(dv)
            imgs_b = torch.stack([tf_bio (img) for img in pil_imgs]).to(dv)

            logits_d = dino(pixel_values=imgs_d).logits
            logits_b = bio(imgs_b)

            probs = softmax(logits_d) * 0.7 + softmax(logits_b) * 0.3   
            preds = probs.argmax(1).cpu().tolist()
            fixed_ids = [Path(a).stem.split('_')[-1]   # keep last chunk
                         for a in ann_ids]             #   e.g. "2_15" → "15"

            rows.extend((aid, idx2cat[p])
                        for aid, p in zip(fixed_ids, preds))


    # ---------- write CSV ----------
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        csv.writer(f).writerow(["annotation_id", "concept_name"])
        csv.writer(f).writerows(rows)

    print(f"Wrote {len(rows)} predictions → {args.out}")

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)   # quiet TQDM pin-mem
    main()