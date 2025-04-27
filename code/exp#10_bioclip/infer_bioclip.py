"""
Inference for BioCLIP fine-tuned checkpoints created by train.py
"""
import argparse, csv, json, os, torch
from pathlib import Path
from torch.utils.data import DataLoader
import open_clip                                      
from dataset import CsvFathomDataset
from tqdm.auto import tqdm

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--test-csv",   required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output",     default="submission.csv")
    p.add_argument("--batch-size", type=int, default=64)
    return p.parse_args()

def device():                      # small helper
    return torch.device("cuda" if torch.cuda.is_available()
                        else "mps" if torch.backends.mps.is_available()
                        else "cpu")

# --------------------------------------------------------------------- #
# Build the exact same wrapper class that was used during training
# --------------------------------------------------------------------- #
class BioCLIPClassifier(torch.nn.Module):
    def __init__(self, clip_backbone, n_cls):
        super().__init__()
        self.backbone = clip_backbone
        self.classifier = torch.nn.Linear(clip_backbone.visual.output_dim, n_cls)
    def forward(self, x):
        feats = self.backbone.encode_image(x)
        return self.classifier(feats)

def main():
    args   = parse_args()
    dev    = device()

    # ---------- 1. label map ----------
    ckpt_dir = Path(args.checkpoint).parent
    with open(ckpt_dir / "label_map.json") as f:
        cat2idx = json.load(f)
    idx2cat = {v:k for k,v in cat2idx.items()}

    # ---------- model (always uses `head.*` to match your checkpoint) ----------
    backbone, _, tf_val = open_clip.create_model_and_transforms(
            'hf-hub:imageomics/bioclip')
    img_dim = backbone.visual.output_dim

    class BioCLIPClassifier(torch.nn.Module):
        def __init__(self, clip_backbone, n_cls):
            super().__init__()
            self.backbone = clip_backbone
            # in your training code this was named `head`
            self.head = torch.nn.Linear(img_dim, n_cls)
        def forward(self, x):
            with torch.no_grad():
                feats = self.backbone.encode_image(x)
            return self.head(feats)

    # load checkpoint (no renaming needed – your saved keys are head.*)
    model = BioCLIPClassifier(backbone, len(cat2idx))
    model.load_state_dict(torch.load(args.checkpoint, map_location='cpu')['model'], strict=True)
    model.to(dev).eval()

    # ---------- 4. data ----------
    ds_test = CsvFathomDataset(args.test_csv, transforms=tf_val)
    loader  = DataLoader(ds_test,
                         batch_size=args.batch_size,
                         shuffle=False, num_workers=2, pin_memory=False)

    # ---------- 5. inference loop ----------
    rows = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Infer"):
            imgs, fnames = batch[0], batch[-1]
            # fnames might be like "23_115.png" → extract "115"
            ann_ids = [ os.path.splitext(f)[0].split('_')[-1] for f in fnames ]
            preds = model(imgs.to(dev)).argmax(1).cpu().tolist()
            rows.extend((aid, idx2cat[p]) for aid, p in zip(ann_ids, preds))

    # ---------- 6. write CSV ----------
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        csv.writer(f).writerow(["annotation_id", "concept_name"])
        csv.writer(f).writerows(rows)

    print(f"wrote {len(rows)} predictions → {args.output}")

if __name__ == "__main__":
    main()