from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import wandb
except ImportError:  # pragma: no cover - wandb optional at runtime
    wandb = None


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_checkpoint(state: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def maybe_init_wandb(enable: bool, project: str, run_name: Optional[str], config: dict):
    if not enable or wandb is None:
        return None
    return wandb.init(project=project, name=run_name, config=config, save_code=False)


def log_images_to_wandb(
    run,
    tag: str,
    samples: Sequence[Tuple[Path, int]],
    class_names: Sequence[str],
    max_images: int = 6,
) -> None:
    if run is None or wandb is None:
        return
    logged = []
    for path, label in samples[:max_images]:
        try:
            from PIL import Image

            image = Image.open(path).convert("RGB")
            caption = f"{path.name} -> {class_names[label]}"
            logged.append(wandb.Image(image, caption=caption))
        except Exception as exc:  # pragma: no cover - defensive logging
            run.log({f"{tag}_error": str(exc)}, commit=False)
    if logged:
        run.log({tag: logged}, commit=False)


def run_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Tuple[float, float]:
    training = optimizer is not None
    model.train(mode=training)
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        if training:
            optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, targets)
        if training:
            loss.backward()
            optimizer.step()
        preds = torch.argmax(logits, dim=1)
        total_loss += loss.item() * inputs.size(0)
        total_correct += (preds == targets).sum().item()
        total_examples += inputs.size(0)
    avg_loss = total_loss / max(total_examples, 1)
    acc = total_correct / max(total_examples, 1)
    return avg_loss, acc


def evaluate_model(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            logits = model(inputs)
            loss = criterion(logits, targets)
            preds = torch.argmax(logits, dim=1)
            total_loss += loss.item() * inputs.size(0)
            total_correct += (preds == targets).sum().item()
            total_examples += inputs.size(0)
    avg_loss = total_loss / max(total_examples, 1)
    acc = total_correct / max(total_examples, 1)
    return avg_loss, acc


def confidence_from_logits(logits: torch.Tensor, method: str = "max") -> torch.Tensor:
    """Convert logits into [0,1]置信度，默认使用 softmax 最大概率。"""
    probs = F.softmax(logits, dim=1)
    if method == "entropy":
        entropy = torch.sum(-probs * torch.log(probs + 1e-8), dim=1)
        max_entropy = torch.log(torch.tensor(probs.shape[1], device=probs.device, dtype=probs.dtype))
        normalized = 1.0 - entropy / (max_entropy + 1e-8)
        return normalized.clamp(0.0, 1.0)
    return probs.max(dim=1).values
