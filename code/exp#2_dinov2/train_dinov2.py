import argparse, os, random, json, time, logging, csv
from pathlib import Path

import torch, numpy as np
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torch import GradScaler, autocast
from transformers import AutoImageProcessor, AutoModelForImageClassification
from sklearn.metrics import precision_recall_fscore_support

from dataset import CsvFathomDataset
from utils_metrics import LabelSmoothingCrossEntropy, accuracy

# ---------- CLI ----------
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train-csv", required=True)
    p.add_argument("--output-dir", default="runs/dinov2_run")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr-head", type=float, default=3e-4)
    p.add_argument("--lr-backbone", type=float, default=3e-5)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--model-id", default="facebook/dinov2-base")
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--workers", type=int, default=None,
                   help="DataLoader workers; default picks sensible value")
    p.add_argument("--debug", action="store_true")
    return p.parse_args()

# ---------- helpers ----------
def pick_device():
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")

def make_logger(out_dir):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(out_dir) / "train.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="w"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger()

# ---------- main ----------
def main():
    args = get_args()
    t0 = time.time()
    device = pick_device()
    logger = make_logger(args.output_dir)
    logger.info(f"Run dir: {args.output_dir}")
    logger.info(f"Device  : {device}")

    random.seed(42); np.random.seed(42); torch.manual_seed(42)

    # 1.  Transforms
    proc = AutoImageProcessor.from_pretrained(args.model_id)
    tf_train = transforms.Compose([
        transforms.RandomResizedCrop((proc.crop_size["height"], proc.crop_size["width"]),
                                     scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(proc.image_mean, proc.image_std),
    ])
    tf_val = transforms.Compose([
        transforms.Resize(proc.size["shortest_edge"]),
        transforms.CenterCrop((proc.crop_size["height"], proc.crop_size["width"])),
        transforms.ToTensor(),
        transforms.Normalize(proc.image_mean, proc.image_std),
    ])

    # 2.  Dataset / split
    ds_all = CsvFathomDataset(args.train_csv, transforms=tf_train)
    if args.debug:
        ds_all.items = ds_all.items[:400]  # quick smoke test
    val_len = max(1, int(0.1 * len(ds_all)))
    train_ds, val_ds = random_split(ds_all, [len(ds_all) - val_len, val_len])
    val_ds.dataset.transforms = tf_val

    slurm_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", 0))
    if args.workers is not None:
        num_workers = args.workers
    elif slurm_cpus:
        num_workers = max(1, slurm_cpus - 1)   # leave 1 CPU for the main proc
    else:
        num_workers = min(8, os.cpu_count() // 4)
    if device.type != "cuda":   # OSC job slot: 1 CPU
        num_workers = 1
    pin = device.type == "cuda"

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=num_workers, pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=num_workers, pin_memory=pin)

    # 3.  Model
    model = AutoModelForImageClassification.from_pretrained(
        args.model_id, num_labels=len(ds_all.cat2idx)
    ).to(device)

    # freeze backbone initially
    for p in model.dinov2.parameters():
        p.requires_grad = False

    # 4.  Optimiser & loss
    # head params = classifier only at first
    head_params = [p for n, p in model.named_parameters() if p.requires_grad]
    opt = torch.optim.AdamW([
        {"params": head_params, "lr": args.lr_head}
    ], weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    criterion = LabelSmoothingCrossEntropy(args.label_smoothing)

    # 5.  TensorBoard & CSV
    from torch.utils.tensorboard import SummaryWriter
    tb = SummaryWriter(log_dir=args.output_dir)
    csv_path = Path(args.output_dir) / "metrics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss",
                         "val_top1", "val_top3", "macro_P", "macro_R", "macro_F1"])

    scaler = GradScaler(enabled=(device.type == "cuda"))

    best_hd = 1e9  # hierarchical distance proxy (lower better)

    # 6.  Training loop
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        for imgs, labels, _ in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            opt.zero_grad()
            with autocast(device_type = "cuda"):
                out = model(pixel_values=imgs)
                loss = criterion(out.logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            running_loss += loss.item() * imgs.size(0)
        train_loss = running_loss / len(train_loader.dataset)

        # ---- UNFREEZE backbone at epoch 4 ----
        if epoch == 4:
            for p in model.dinov2.parameters():
                p.requires_grad = True
            opt = torch.optim.AdamW([
                {"params": model.classifier.parameters(), "lr": args.lr_head},
                {"params": model.dinov2.parameters(), "lr": args.lr_backbone}
            ], weight_decay=args.weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=args.epochs - epoch + 1)
            logger.info("Backbone unfrozen with lr {:.1e}".format(args.lr_backbone))

        # ---- Validation ----
        model.eval()
        val_loss = 0.0
        correct1 = correct3 = 0
        preds_all, labels_all = [], []
        with torch.no_grad():
            for imgs, labels, _ in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                with autocast(device_type = "cuda"):
                    out = model(pixel_values=imgs)
                    loss = criterion(out.logits, labels)
                val_loss += loss.item() * imgs.size(0)

                top1, top3 = accuracy(out.logits, labels, topk=(1, 3))
                correct1 += top1.item() * imgs.size(0) / 100
                correct3 += top3.item() * imgs.size(0) / 100

                preds_all.append(out.logits.argmax(1).cpu())
                labels_all.append(labels.cpu())

        val_loss /= len(val_loader.dataset)
        val_top1 = correct1 / len(val_loader.dataset)
        val_top3 = correct3 / len(val_loader.dataset)
        y_pred = torch.cat(preds_all).numpy()
        y_true = torch.cat(labels_all).numpy()
        P, R, F1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0)

        # ---- Logging ----
        tb.add_scalar("Loss/train", train_loss, epoch)
        tb.add_scalar("Loss/val",   val_loss,   epoch)
        tb.add_scalar("Acc/top1",   val_top1,   epoch)
        tb.add_scalar("Acc/top3",   val_top3,   epoch)
        tb.add_scalar("MacroF1",    F1,         epoch)
        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, train_loss, val_loss,
                                    val_top1, val_top3, P, R, F1])

        logger.info(
            f"Epoch {epoch:02d} | "
            f"train_loss {train_loss:.4f} | "
            f"val_loss {val_loss:.4f} | "
            f"top1 {val_top1:.3f} | top3 {val_top3:.3f} | "
            f"P/R/F1 {P:.3f}/{R:.3f}/{F1:.3f} | "
            f"epoch_time {(time.time()-epoch_start):.1f}s"
        )

        # ---- Checkpoint ----
        torch.save(model.state_dict(), f"{args.output_dir}/epoch_{epoch:02d}.pth")

        # after we re‑create the scheduler at epoch 4
        # we must skip stepping it until the *next* epoch
        if not (epoch == 4):
            scheduler.step()

    total_time = time.time() - t0
    logger.info(f"Training complete in {total_time/60:.1f} min")
    tb.close()

    # Also persist label map
    with open(Path(args.output_dir) / "label_map.json", "w") as f:
        json.dump(ds_all.cat2idx, f)

if __name__ == "__main__":
    main()