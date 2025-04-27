"""
Unified training script for FathomNet 2025
Handles both HuggingFace vision models (e.g. facebook/dinov2-base)
and OpenCLIP/BioCLIP via --model_backend openclip
"""
import argparse, json, logging, os, random, time, csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torch import optim
from torch.amp import GradScaler, autocast
from sklearn.metrics import precision_recall_fscore_support

from dataset import CsvFathomDataset
from utils_metrics import accuracy, LabelSmoothingCrossEntropy

# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train-csv", required=True)
    p.add_argument("--output-dir", default="runs/exp_dino")
    p.add_argument("--model-backend", choices=["hf", "openclip"], default="hf",
                   help="'hf' = Transformers,  'openclip' = OpenCLIP/BioCLIP")
    p.add_argument("--model-id",
                   default="facebook/dinov2-base",
                   help="HF model name or open_clip name")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--patience", type=int, default=5,
                   help="Early-stop patience on val_top1 (0 disables)")
    p.add_argument("--lr-head", type=float, default=3e-4)
    p.add_argument("--lr-backbone", type=float, default=3e-5)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-every", type=int, default=5)
    p.add_argument("--resume", type=str, default=None,
                   help="Path to checkpoint.pth to resume")
    p.add_argument("--debug", action="store_true")
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_logger(out_dir):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        handlers=[
            logging.FileHandler(Path(out_dir) / "train.log", mode="w"),
            logging.StreamHandler()
        ],
    )
    return logging.getLogger()


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    args = parse_args()
    set_seed(args.seed)
    device = pick_device()
    logger = make_logger(args.output_dir)
    logger.info(f"Device: {device}")

    # ------------------------- dataset & transforms ------------------------ #
    if args.model_backend == "hf":
        from transformers import AutoImageProcessor
        proc = AutoImageProcessor.from_pretrained(args.model_id)
        crop_sz = proc.crop_size["height"]
        resize_short = proc.size["shortest_edge"]
        norm_mean, norm_std = proc.image_mean, proc.image_std
    else:  # openclip
        import open_clip
        _, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
            args.model_id)
        # open_clip returns transforms directly
        tf_train, tf_val = preprocess_train, preprocess_val

    if args.model_backend == "hf":
        tf_train = transforms.Compose([
            transforms.RandomResizedCrop((crop_sz, crop_sz), scale=(0.75, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std),
        ])
        tf_val = transforms.Compose([
            transforms.Resize(resize_short),
            transforms.CenterCrop((crop_sz, crop_sz)),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std),
        ])

    ds_all = CsvFathomDataset(args.train_csv, transforms=tf_train)
    if args.debug:
        ds_all.items = ds_all.items[:400]
    val_len = max(1, int(0.1 * len(ds_all)))
    train_ds, val_ds = random_split(ds_all, [len(ds_all) - val_len, val_len])
    val_ds.dataset.transforms = tf_val

    slurm_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", 0))
    workers = (
        args.workers
        if args.workers is not None
        else max(1, (slurm_cpus - 1) if slurm_cpus else os.cpu_count() // 4)
    )
    if device.type != "cuda":
        workers = 1
    pin_memory = device.type == "cuda"

    loader_train = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=pin_memory,
    )
    loader_val = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin_memory,
    )

    # ------------------------- model ------------------------ #
    if args.model_backend == "hf":
        from transformers import AutoModelForImageClassification
        model = AutoModelForImageClassification.from_pretrained(
            args.model_id, num_labels=len(ds_all.cat2idx)
        )
    else:
        import open_clip
        model, _, _ = open_clip.create_model_and_transforms(
            args.model_id, pretrained=True, num_classes=len(ds_all.cat2idx)
        )
    model.to(device)

    # freeze backbone
    if hasattr(model, "vision_model"):  # HF ViT family
        backbone_params = model.vision_model.parameters()
    elif hasattr(model, "transformer"):  # CLIP
        backbone_params = model.transformer.parameters()
    else:
        backbone_params = []
    for p in backbone_params:
        p.requires_grad = False

    head_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(
        [{"params": head_params, "lr": args.lr_head}], weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler(enabled=(device.type == "cuda"), init_scale=2.**8, growth_interval=2000)
    criterion = LabelSmoothingCrossEntropy(args.label_smoothing)

    start_epoch = 1
    best_top1 = 0.0

    # ------------------------- (optional) resume --------------------------- #
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        best_top1 = ckpt["best_top1"]
        logger.info(f"✓ Resumed from {args.resume} (epoch {ckpt['epoch']})")

    # --------------- logging CSV + TensorBoard ---------------------------- #
    from torch.utils.tensorboard import SummaryWriter

    tb = SummaryWriter(args.output_dir)
    csv_path = Path(args.output_dir) / "metrics.csv"
    if not args.resume:
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(
                ["epoch", "train_loss", "val_loss", "val_top1", "val_top3"]
            )

    patience_cnt = 0
    t0 = time.time()

    # ---------------------------------------------------------------------- #
    # Training loop
    # ---------------------------------------------------------------------- #
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        epoch_t0 = time.time()
        train_loss = 0.0

        for imgs, labels, _ in loader_train:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            with autocast(device_type=device.type):
                out = model(pixel_values=imgs) if args.model_backend == "hf" else model(imgs)
                loss = criterion(out.logits, labels)
            if not torch.isfinite(loss):
                logger.warning("Non-finite loss, batch skipped")
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * imgs.size(0)

        train_loss /= len(loader_train.dataset)

        # ------- unfreeze at epoch 4 ------------------------------------- #
        if epoch == 4 and start_epoch <= 4:
            for p in backbone_params:
                p.requires_grad = True
            optimizer = optim.AdamW(
                [
                    {"params": backbone_params, "lr": args.lr_backbone},
                    {"params": model.classifier.parameters(), "lr": args.lr_head},
                ],
                weight_decay=args.weight_decay,
            )
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs - epoch + 1
            )
            logger.info("Backbone unfrozen")

        # ---------------------- validation ------------------------------- #
        model.eval()
        val_loss = 0.0
        correct1 = correct3 = 0
        preds_cat, labels_cat = [], []
        with torch.no_grad():
            for imgs, labels, _ in loader_val:
                imgs, labels = imgs.to(device), labels.to(device)
                with autocast(device_type=device.type):
                    out = model(pixel_values=imgs) if args.model_backend == "hf" else model(imgs)
                    loss = criterion(out.logits, labels)
                val_loss += loss.item() * imgs.size(0)

                t1, t3 = accuracy(out.logits, labels, topk=(1, 3))
                correct1 += t1.item() * imgs.size(0) / 100
                correct3 += t3.item() * imgs.size(0) / 100
                preds_cat.append(out.logits.argmax(1).cpu())
                labels_cat.append(labels.cpu())

        val_loss /= len(loader_val.dataset)
        val_top1 = correct1 / len(loader_val.dataset)
        val_top3 = correct3 / len(loader_val.dataset)

        # ---------------------- Logging ---------------------------------- #
        tb.add_scalar("Loss/train", train_loss, epoch)
        tb.add_scalar("Loss/val", val_loss, epoch)
        tb.add_scalar("Acc/val_top1", val_top1, epoch)

        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, train_loss, val_loss, val_top1, val_top3])

        logger.info(
            f"Ep{epoch:02d}  loss {train_loss:.4f}/{val_loss:.4f}  "
            f"top1 {val_top1:.3f}  top3 {val_top3:.3f}  "
            f"time {(time.time()-epoch_t0):.1f}s"
        )

        # ---------------------- checkpointing ---------------------------- #
        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_top1": best_top1,
        }
        if epoch % args.save_every == 0:
            torch.save(ckpt, Path(args.output_dir) / f"epoch_{epoch}.pth")

        if val_top1 > best_top1:              # best model
            best_top1 = val_top1
            torch.save(ckpt, Path(args.output_dir) / "best.pth")
            patience_cnt = 0
        else:
            patience_cnt += 1

        # ---------------------- LR schedule ------------------------------ #
        scheduler.step()

        # ---------------------- early stopping --------------------------- #
        if args.patience and patience_cnt >= args.patience:
            logger.info(f"Early stopping after {patience_cnt} epochs without improvement.")
            break

    logger.info(f"Training done in {(time.time()-t0)/60:.1f} min")
    tb.close()

    # label map for inference script
    with open(Path(args.output_dir) / "label_map.json", "w") as f:
        json.dump(ds_all.cat2idx, f)


if __name__ == "__main__":
    main()