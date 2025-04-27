import argparse, csv, json, os, torch
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import AutoImageProcessor, AutoModelForImageClassification
from dataset import CsvFathomDataset
from tqdm.auto import tqdm

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--test-csv',   required=True,
                   help='Path to your test CSV (no labels)')
    p.add_argument('--checkpoint', required=True,
                   help='Path to the .pth you saved (best_model.pth)')
    p.add_argument('--output',     default='submission.csv',
                   help='Where to write your predictions')
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--model-id',   default='facebook/dinov2-base')
    return p.parse_args()

def pick_device():
    if torch.backends.mps.is_available(): return torch.device('mps')
    if torch.cuda.is_available():         return torch.device('cuda')
    return torch.device('cpu')

def main():
    args = parse_args()
    device = pick_device()

    # 1) Load label map from same folder as checkpoint
    ckpt_dir = os.path.dirname(args.checkpoint)
    label_map_path = os.path.join(ckpt_dir, 'label_map.json')
    if not os.path.exists(label_map_path):
        raise FileNotFoundError(f"Can't find label_map.json at {label_map_path}")
    with open(label_map_path) as f:
        cat2idx = json.load(f)
    idx2cat = {v:k for k,v in cat2idx.items()}

    # 2) Build transforms
    proc = AutoImageProcessor.from_pretrained(args.model_id)
    tf = transforms.Compose([
        transforms.Resize(proc.size['shortest_edge']),
        transforms.CenterCrop((proc.crop_size['height'], proc.crop_size['width'])),
        transforms.ToTensor(),
        transforms.Normalize(proc.image_mean, proc.image_std),
    ])

    # 3) Load test dataset (CsvFathomDataset should return (img, ann_id) or (img,_,ann_id))
    ds_raw = CsvFathomDataset(args.test_csv, transforms=tf)
    loader = DataLoader(ds_raw,
                        batch_size=args.batch_size,
                        shuffle=False,
                        num_workers=0,
                        pin_memory=False)

    # 4) Build model with the right number of classes
    num_labels = len(cat2idx)
    model = AutoModelForImageClassification.from_pretrained(
        args.model_id, num_labels=num_labels
    )
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device).eval()

    # 5) Inference loop
    rows = []
    with torch.no_grad():
        for batch in tqdm(loader, desc='Infer'):
            # unpack batch of size 2 or 3
            if len(batch) == 2:
                imgs, ann_ids = batch
            elif len(batch) == 3:
                imgs, _, ann_ids = batch
            else:
                raise ValueError(f'Unexpected batch size: {len(batch)}')
            imgs = imgs.to(device)
            preds = model(pixel_values=imgs).logits.argmax(1).cpu().tolist()
            for ann, p in zip(ann_ids, preds):
                rows.append((ann, idx2cat[p]))

    # 6) Write submission
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['annotation_id', 'concept_name'])
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows → {args.output}")

if __name__ == '__main__':
    main()