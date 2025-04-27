import argparse, os, random, json, torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, models
from tqdm.auto import tqdm
from transformers import AutoImageProcessor, AutoModelForImageClassification
from sklearn.metrics import precision_recall_fscore_support
from torch.nn.utils import clip_grad_norm_
from dataset import CsvFathomDataset

# ---------- CLI ----------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train-csv", default="dataset/dataset/train/annotations.csv")
    p.add_argument("--output-dir", default="outputs")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--model-id", default="microsoft/resnet-18")
    p.add_argument("--debug", action="store_true")
    return p.parse_args()

# ---------- util ----------
def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

# ---------- main ----------
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    random.seed(42); torch.manual_seed(42)
    device = pick_device()

    crop_size = 224

    # -------- transforms --------
    proc = AutoImageProcessor.from_pretrained(args.model_id)
    tf_train = transforms.Compose([
        transforms.RandomResizedCrop((crop_size, crop_size), scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(proc.image_mean, proc.image_std),
    ])
    tf_val = transforms.Compose([
        transforms.Resize(proc.size["shortest_edge"]),
        transforms.CenterCrop((crop_size, crop_size)),
        transforms.ToTensor(),
        transforms.Normalize(proc.image_mean, proc.image_std),
    ])

    # -------- dataset --------
    ds = CsvFathomDataset(args.train_csv, transforms=tf_train)
    if args.debug:
        ds.items = ds.items[:100]
    val_n = max(1, int(len(ds) * 0.1))
    train_ds, val_ds = random_split(ds, [len(ds) - val_n, val_n])
    val_ds.dataset.transforms = tf_val

    num_workers = 0 if device.type in ("cpu", "mps") else 4
    pin_memory  = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin_memory)

    # -------- model --------
    num_labels = len(ds.cat2idx)
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.classifier = torch.nn.Linear(model.fc.in_features, num_labels)
    model = model.to(device)


    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()
    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):

        # ------- train -------
        model.train()
        train_loss = 0.0
        for imgs, labels, _ in tqdm(train_loader, desc=f"Epoch {epoch} · train", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # ------- validate -------
        model.eval()
        val_loss = correct = total = 0.0
        preds_all, labels_all = [], []
        with torch.no_grad():
            for imgs, labels, _ in tqdm(val_loader, desc=f"Epoch {epoch} · val", leave=False):
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                val_loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total   += labels.size(0)
                preds_all.append(preds.cpu())
                labels_all.append(labels.cpu())
        val_loss /= len(val_loader)
        val_acc = correct / total
        y_true = torch.cat(labels_all).numpy()
        y_pred = torch.cat(preds_all).numpy()
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)

        print(f"Epoch {epoch:02d} | train_loss {train_loss:.4f} | "
              f"val_loss {val_loss:.4f} | val_acc {val_acc:.4f} | "
              f"P/R/F1 {p:.3f}/{r:.3f}/{f1:.3f}")

        # best & last checkpoints
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model.pth"))
        torch.save(model.state_dict(), os.path.join(args.output_dir, f"epoch_{epoch}.pth"))
        scheduler.step()

    with open(os.path.join(args.output_dir, "label_map.json"), "w") as f:
        json.dump(ds.cat2idx, f)

if __name__ == "__main__":
    main()