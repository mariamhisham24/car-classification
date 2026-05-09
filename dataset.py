"""
dataset.py
==========
PyTorch Dataset and DataLoader factory for the car classification task.
Applies full ImageNet-style preprocessing + data augmentation for training.
"""

import os
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


# ---------------------------------------------------------------------------
# Class mapping (discovered automatically from folder structure)
# ---------------------------------------------------------------------------

def discover_classes(root: str) -> dict:
    """Return sorted {class_name: idx} from sub-folders of `root`."""
    classes = sorted(
        d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))
    )
    return {cls: i for i, cls in enumerate(classes)}


# ---------------------------------------------------------------------------
# Transform pipelines
# ---------------------------------------------------------------------------

IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD  = [0.229, 0.224, 0.225]

def get_train_transforms(img_size: int = 224) -> transforms.Compose:
    """
    Full augmented pipeline for training:
      1.  Resize (slightly larger than crop target)
      2.  Random Horizontal Flip
      3.  Random Rotation ±20°
      4.  Random Resized Crop → img_size
      5.  Color Jitter (brightness, contrast, saturation, hue)
      6.  Random Perspective
      7.  Random Grayscale (5 % chance – forces model to be colour-agnostic)
      8.  ToTensor (HWC uint8 → CHW float [0,1])
      9.  ImageNet Normalization (subtract mean, divide std)
      10. Random Erasing (simulates occlusion)
    """
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.15)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=20),
        transforms.RandomResizedCrop(img_size, scale=(0.75, 1.0)),
        transforms.ColorJitter(brightness=0.4, contrast=0.4,
                               saturation=0.4, hue=0.1),
        transforms.RandomPerspective(distortion_scale=0.3, p=0.4),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMG_MEAN, std=IMG_STD),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),
    ])


def get_val_transforms(img_size: int = 224) -> transforms.Compose:
    """
    Deterministic pipeline for validation / test:
      1.  Resize to img_size (no crop, uses full image via pad-then-resize)
      2.  Center Crop
      3.  ToTensor
      4.  ImageNet Normalization
    """
    return transforms.Compose([
        transforms.Resize(img_size + 32),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMG_MEAN, std=IMG_STD),
    ])


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CarDataset(Dataset):
    """
    Folder-based car image dataset.
    Expected structure:
        root/
          audi/         *.jpg
          lamborghini/  *.jpg
          mercedes/     *.jpg
    """

    EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(self, root: str, transform=None, class_to_idx: dict = None):
        self.root      = Path(root)
        self.transform = transform

        if class_to_idx is None:
            class_to_idx = discover_classes(root)
        self.class_to_idx = class_to_idx
        self.idx_to_class = {v: k for k, v in class_to_idx.items()}

        self.samples: list[Tuple[Path, int]] = []
        for cls, idx in class_to_idx.items():
            cls_dir = self.root / cls
            if not cls_dir.exists():
                continue
            for f in sorted(cls_dir.iterdir()):
                if f.suffix.lower() in self.EXTS:
                    self.samples.append((f, idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

    def class_counts(self) -> dict:
        counts = {}
        for _, lbl in self.samples:
            cls = self.idx_to_class[lbl]
            counts[cls] = counts.get(cls, 0) + 1
        return counts


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def get_dataloaders(
    data_dir: str,
    img_size: int = 224,
    batch_size: int = 16,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, dict, int]:
    """
    Returns (train_loader, val_loader, class_to_idx, num_classes).
    """
    train_root = os.path.join(data_dir, "Train")
    test_root  = os.path.join(data_dir, "Test")

    class_to_idx = discover_classes(train_root)
    num_classes  = len(class_to_idx)

    train_ds = CarDataset(train_root, get_train_transforms(img_size), class_to_idx)
    val_ds   = CarDataset(test_root,  get_val_transforms(img_size),   class_to_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers,
                              pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size,
                              shuffle=False, num_workers=num_workers,
                              pin_memory=True)

    print(f"[dataset] Classes  : {list(class_to_idx.keys())}")
    print(f"[dataset] Train set: {len(train_ds)} images")
    print(f"[dataset] Val set  : {len(val_ds)}   images")
    print(f"[dataset] Train class distribution: {train_ds.class_counts()}")

    return train_loader, val_loader, class_to_idx, num_classes
