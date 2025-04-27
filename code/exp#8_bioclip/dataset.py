"""
CsvFathomDataset
----------------
Very small utility that reads the competition-style CSV and serves PIL images
together with integer labels.  It works for both train *and* test:

train CSV header:  path,label
test  CSV header:  path            (no label column)

Returned tuples
---------------

* training / validation:  (Tensor image, int label_idx, str annotation_id)
* test / inference:       (Tensor image, str annotation_id)

`annotation_id` is the basename of the file **without** extension, e.g.
IMG_12345.jpg → 12345.  Feel free to adapt if your ID rule is different.
"""
from pathlib import Path
import csv, os
from PIL import Image

from torch.utils.data import Dataset


class CsvFathomDataset(Dataset):
    def __init__(self, csv_path: str, transforms=None) -> None:
        self.items = []                      # list[(img_path:str, label:str|None)]
        self.transforms = transforms

        self.label2idx = {}                  # str label → int index
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            has_label = 'label' in reader.fieldnames

            for row in reader:
                img_path = row['path']
                lbl      = row.get('label') if has_label else None
                self.items.append((img_path, lbl))

                if has_label:
                    # assign a new integer to each *new* category we meet
                    if lbl not in self.label2idx:
                        self.label2idx[lbl] = len(self.label2idx)

        # expose immutable mapping for outside code
        self.cat2idx = self.label2idx

    # ------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.items)

    # ------------------------------------------------------------
    def __getitem__(self, idx: int):
        img_path, lbl = self.items[idx]

        # Always convert to RGB so we have 3 channels
        img = Image.open(img_path).convert("RGB")
        if self.transforms is not None:
            img = self.transforms(img)

        # Use filename (without extension) as annotation_id
        ann_id = Path(img_path).stem

        if lbl is None:                      # inference mode
            return img, ann_id

        label_idx = self.label2idx[lbl]
        return img, label_idx, ann_id