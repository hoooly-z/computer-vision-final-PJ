from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset

from . import features

CLASSES: Tuple[str, ...] = ("fire", "start_fire", "no_fire")
CLASS_TO_IDX = {cls_name: idx for idx, cls_name in enumerate(CLASSES)}
IDX_TO_CLASS = {idx: cls for cls, idx in CLASS_TO_IDX.items()}
SUBMISSION_LABELS = {"fire": 0, "no_fire": 1, "start_fire": 2}
IDX_TO_SUBMISSION = {CLASS_TO_IDX[name]: SUBMISSION_LABELS[name] for name in CLASSES}


def list_image_files(root: Path, classes: Sequence[str] = CLASSES) -> List[Tuple[Path, int]]:
    """Collect image paths and integer labels under root/class folders."""
    samples: List[Tuple[Path, int]] = []
    for cls in classes:
        class_dir = root / cls
        if not class_dir.exists():
            continue
        for path in sorted(class_dir.rglob("*")):
            if path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                samples.append((path, CLASS_TO_IDX[cls]))
    return samples


class FeatureDataset(Dataset):
    """Dataset that returns cached histogram or FFT features."""

    def __init__(
        self,
        root: Path,
        feature_type: str,
        cache_dir: Optional[Path] = None,
        split: str = "train",
    ):
        self.root = Path(root)
        self.feature_type = feature_type
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.split = split
        self.samples = list_image_files(self.root)
        if feature_type not in {"hist", "fft"}:
            raise ValueError(f"Unsupported feature type: {feature_type}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        if self.feature_type == "hist":
            vec = features.load_hist_features(path, cache_dir=self.cache_dir, split=self.split)
        else:
            vec = features.load_fft_features(path, cache_dir=self.cache_dir, split=self.split)
        tensor = torch.from_numpy(vec)
        return tensor, label


class ImageDataset(Dataset):
    """Standard image dataset for CNN training."""

    def __init__(self, root: Path, transform: Optional[Callable] = None, return_path: bool = False):
        self.root = Path(root)
        self.transform = transform
        self.return_path = return_path
        self.samples = list_image_files(self.root)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        if self.return_path:
            return image, label, path
        return image, label


class TestImageDataset(Dataset):
    """Dataset for unlabeled inference images."""

    def __init__(self, root: Path, transform: Optional[Callable] = None):
        self.root = Path(root)
        self.transform = transform
        self.paths = sorted(
            path for path in self.root.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, path.name, path
