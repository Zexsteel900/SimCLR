"""
dataset.py
EuroSAT dataset loading and SimCLR-style augmentation pipeline.
"""

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
import numpy as np


# ------------------------------------------------------------------
# SimCLR Augmentation Pipeline
# ------------------------------------------------------------------

class SimCLRTransform:
    """
    Produces two randomly augmented views of the same image,
    following the SimCLR augmentation strategy (Chen et al., 2020).
    Applied during self-supervised pre-training.
    """

    def __init__(self, image_size: int = 64):
        color_jitter = transforms.ColorJitter(
            brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
        )
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(size=image_size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=int(0.1 * image_size) | 1),  # odd kernel
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.3444, 0.3803, 0.4078],
                                 std=[0.2034, 0.1365, 0.1148]),  # EuroSAT RGB stats
        ])

    def __call__(self, x):
        return self.transform(x), self.transform(x)


class EvalTransform:
    """Standard deterministic transform for fine-tuning and evaluation."""

    def __init__(self, image_size: int = 64):
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.3444, 0.3803, 0.4078],
                                 std=[0.2034, 0.1365, 0.1148]),
        ])

    def __call__(self, x):
        return self.transform(x)


# ------------------------------------------------------------------
# Dataset helpers
# ------------------------------------------------------------------

def get_eurosat_pretrain(data_root: str = "./data", batch_size: int = 256,
                         num_workers: int = 4):
    """
    Returns a DataLoader over the full EuroSAT dataset (all 27,000 images)
    with SimCLR dual-view augmentations for self-supervised pre-training.
    """
    dataset = datasets.EuroSAT(
        root=data_root, download=True,
        transform=SimCLRTransform(image_size=64)
    )
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    return loader


def get_eurosat_finetune(data_root: str = "./data", label_fraction: float = 0.10,
                         batch_size: int = 128, num_workers: int = 4,
                         seed: int = 42):
    """
    Returns train and test DataLoaders for the labeled fine-tuning stage.
    label_fraction: fraction of the training split to use as labeled data (default 10%).
    The split is stratified so each class is proportionally represented.
    """
    full_dataset = datasets.EuroSAT(
        root=data_root, download=True,
        transform=EvalTransform(image_size=64)
    )

    targets = np.array(full_dataset.targets)
    indices = np.arange(len(targets))

    # 80/20 train-test split
    train_idx, test_idx = train_test_split(
        indices, test_size=0.20, stratify=targets, random_state=seed
    )

    # Sub-sample labeled training data
    train_targets = targets[train_idx]
    labeled_idx, _ = train_test_split(
        train_idx, train_size=label_fraction,
        stratify=train_targets, random_state=seed
    )

    train_loader = DataLoader(
        Subset(full_dataset, labeled_idx),
        batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        Subset(full_dataset, test_idx),
        batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    print(f"  Labeled training samples : {len(labeled_idx)}")
    print(f"  Test samples             : {len(test_idx)}")
    return train_loader, test_loader
