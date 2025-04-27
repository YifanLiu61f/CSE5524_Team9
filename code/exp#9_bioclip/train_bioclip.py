import argparse, json, logging, os, random, time, csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torch import optim, nn
from torch import GradScaler, autocast
from sklearn.metrics import precision_recall_fscore_support

# ---------- 1. local ----------
from dataset import CsvFathomDataset               
from utils_metrics import accuracy, LabelSmoothingCrossEntropy


# ---------- 2. CLI ----------
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--train-csv',  required=True)
    p.add_argument('--out',        default='runs/bioclip_run')
    p.add_argument('--batch',      type=int, default=32)
    p.add_argument('--epochs',     type=int, default=30)
    p.add_argument('--freeze',     type=int, default=5)
    p.add_argument('--lr-head',    type=float, default=1e-3)
    p.add_argument('--lr-back',    type=float, default=3e-5)
    p.add_argument('--wd',         type=float, default=1e-2)
    p.add_argument('--label-smooth', type=float, default=0.1)
    p.add_argument('--workers',    type=int,   default=4)
    p.add_argument('--seed',       type=int,   default=42)
    p.add_argument('--resume',     type=str)
    return p.parse_args()


# ---------- 3. helpers ----------
def seed_everywhere(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def pick_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def logger_for(out_dir: str):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(message)s',
        handlers=[logging.FileHandler(Path(out_dir)/'train.log', 'w'),
                  logging.StreamHandler()]
    )
    return logging.getLogger()


# ---------- 4. main ----------
def main():
    args   = get_args()
    device = pick_device()
    seed_everywhere(args.seed)
    log = logger_for(args.out)
    log.info(f'Device: {device}')

    # ---- 4-1 load BioCLIP ----
    import open_clip
    backbone, tf_train_clip, tf_val_clip = open_clip.create_model_and_transforms(
        'hf-hub:imageomics/bioclip')
    img_dim = backbone.visual.output_dim            # 512

    # ---------- extra data aug ---------
    extra_aug = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.2,0.2,0.2,0.1)], p=0.8),
        transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.1,2.0))], p=0.5),
    ])
    tf_train = transforms.Compose([extra_aug, tf_train_clip])   # ◀ stack
    tf_val   = tf_val_clip        # unchanged

    # ---- 4-2 classifier head ----
    class BioCLIPClassifier(nn.Module):
        def __init__(self, clip_backbone, n_cls):
            super().__init__()
            self.backbone = clip_backbone
            self.head     = nn.Linear(img_dim, n_cls)    
        def forward(self, x):
            feats = self.backbone.encode_image(x)    
            return self.head(feats)

    # ---- 4-3 data ----
    ds_full = CsvFathomDataset(args.train_csv, transforms=tf_train)
    val_len = max(1, int(0.1*len(ds_full)))
    ds_train, ds_val = random_split(ds_full, [len(ds_full)-val_len, val_len])
    ds_val.dataset.transforms = tf_val      

    train_ld = DataLoader(ds_train, batch_size=args.batch, shuffle=True,
                          num_workers=args.workers, pin_memory=True)
    val_ld   = DataLoader(ds_val,   batch_size=args.batch, shuffle=False,
                          num_workers=args.workers, pin_memory=True)

    model = BioCLIPClassifier(backbone, len(ds_full.cat2idx)).to(device)

    # freeze backbone params
    for p in model.backbone.parameters():
        p.requires_grad_(False)

    criterion = LabelSmoothingCrossEntropy(args.label_smooth)
    scaler    = GradScaler(enabled=device.type=='cuda')

    # ---------- optimiser / sched  ----------
    opt   = optim.AdamW(model.head.parameters(),
                        lr=args.lr_head,           # 1 e-3 OK for head
                        weight_decay=1e-2)         # ↓ WD
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)


    start_ep, best_top1 = 1, 0.0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model'])
        opt.load_state_dict(ckpt['opt'])
        sched.load_state_dict(ckpt['sched'])
        scaler.load_state_dict(ckpt['scaler'])
        start_ep   = ckpt['ep'] + 1
        best_top1  = ckpt['best']
        log.info(f'✓ resume from {args.resume}')

    # ---- 4-4 training ----
    csv_path = Path(args.out)/'metrics.csv'
    if start_ep == 1:             # fresh run
        with open(csv_path,'w',newline='') as f:
            csv.writer(f).writerow(['ep','tr_loss','val_loss','val_top1'])

    for ep in range(start_ep, args.epochs+1):
        t0 = time.time()
        model.train()
        tr_loss = 0.
        for imgs, lbls, _ in train_ld:
            imgs, lbls = imgs.to(device), lbls.to(device)
            opt.zero_grad()
            with autocast(device_type = "cuda"):
                logits = model(imgs)
                loss = criterion(logits, lbls)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            tr_loss += loss.item()*imgs.size(0)
        tr_loss /= len(ds_train)

        # ---------- unfreeze ----------
        if ep == args.freeze+1:
            for p in model.backbone.parameters():
                p.requires_grad_(True)
            opt = optim.AdamW([
                {'params': model.head.parameters(),     'lr': args.lr_head},
                {'params': model.backbone.parameters(), 'lr': args.lr_back}
            ], weight_decay=args.wd)
            sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs-ep+1)
            log.info(f'Backbone unfrozen @epoch{ep}')

        # ---------- evaluation ----------
        model.eval()
        val_loss, corr1 = 0., 0.
        with torch.no_grad():
            for imgs, lbls, _ in val_ld:
                imgs, lbls = imgs.to(device), lbls.to(device)
                with autocast(device_type = "cuda"):
                    logits = model(imgs)
                    loss   = criterion(logits, lbls)
                val_loss += loss.item()*imgs.size(0)
                corr1    += (logits.argmax(1)==lbls).sum().item()
        val_loss /= len(ds_val)
        top1 = corr1 / len(ds_val)

        # ---------- log ----------
        log.info(f'Ep{ep:02d}  loss {tr_loss:.3f}/{val_loss:.3f}  top1 {top1:.3f}  t {time.time()-t0:.1f}s')
        with open(csv_path,'a',newline='') as f:
            csv.writer(f).writerow([ep,tr_loss,val_loss,top1])

        # ---------- save ----------
        ckpt = {'ep':ep,'model':model.state_dict(),'opt':opt.state_dict(),
                'sched':sched.state_dict(),'scaler':scaler.state_dict(),'best':best_top1}
        torch.save(ckpt, Path(args.out)/'last.pth')
        if top1>best_top1:
            best_top1 = top1
            torch.save(ckpt, Path(args.out)/'best.pth')

        sched.step()

    # label-map
    with open(Path(args.out)/'label_map.json','w') as f:
        json.dump(ds_full.cat2idx, f)


if __name__ == '__main__':
    main()