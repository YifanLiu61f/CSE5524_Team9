import csv, os
from PIL import Image
from torch.utils.data import Dataset

class CsvFathomDataset(Dataset):
    """
    If the CSV has a header `path,label` the loader
    1)  reads every row
    2)  builds a label->idx map on the fly
    For the test split (no label column) just header `path`
    """
    def __init__(self, csv_path, transforms=None):
        self.items = []
        self.transforms = transforms
        self.label2idx = {}
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            has_label = 'label' in reader.fieldnames
            for row in reader:
                img_path = row['path']
                lbl      = row.get('label')
                self.items.append((img_path, lbl))
                if has_label:
                    self.label2idx.setdefault(lbl, len(self.label2idx))
        # freeze cat2idx for outside use
        self.cat2idx = self.label2idx

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_path, lbl = self.items[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transforms:
            img = self.transforms(img)
        if lbl is None:          # test time
            return img, os.path.basename(img_path)  # annotation_id fallback
        label_idx = self.label2idx[lbl]
        ann_id    = os.path.splitext(os.path.basename(img_path))[0].split('_')[-1]
        return img, label_idx, ann_id