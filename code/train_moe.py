from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from config_utils import parse_args_with_config
from datasets import CLASSES, ImageDataset
from features import FFT_FEATURE_DIM, HIST_FEATURE_DIM, load_fft_features, load_hist_features
from models import ExpertMoE, FFTMLP, HistMLP, SmallFireCNN
from utils import confidence_from_logits, get_device, maybe_init_wandb, save_checkpoint, set_seed

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA_ROOT = BASE_DIR / "dataset"
DEFAULT_CACHE_DIR = BASE_DIR / ".feature_cache"
DEFAULT_ARTIFACTS = BASE_DIR / "artifacts"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train mixture-of-experts gate.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--hist-checkpoint", type=Path, default=DEFAULT_ARTIFACTS / "hist_expert.pth")
    parser.add_argument("--fft-checkpoint", type=Path, default=DEFAULT_ARTIFACTS / "fft_expert.pth")
    parser.add_argument("--cnn-checkpoint", type=Path, default=DEFAULT_ARTIFACTS / "cnn_expert.pth")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="fire-moe")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--image-size", type=int, default=224)
    return parser


def parse_args() -> argparse.Namespace:
    parser = build_parser()
    return parse_args_with_config(parser)


def load_checkpoint(model: nn.Module, ckpt_path: Path, device: torch.device) -> None:
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])


def compute_feature_batches(paths: List[Path], cache_dir: Path, split: str) -> Tuple[torch.Tensor, torch.Tensor]:
    hist_vectors = [load_hist_features(path, cache_dir=cache_dir, split=split) for path in paths]
    fft_vectors = [load_fft_features(path, cache_dir=cache_dir, split=split) for path in paths]
    hist_tensor = torch.from_numpy(np.stack(hist_vectors)).float()
    fft_tensor = torch.from_numpy(np.stack(fft_vectors)).float()
    return hist_tensor, fft_tensor


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    cache_dir = args.cache_dir

    transform = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
        ]
    )
    train_dataset = ImageDataset(args.data_root / "train", transform=transform, return_path=True)
    val_dataset = ImageDataset(args.data_root / "val", transform=transform, return_path=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    hist_model = HistMLP(HIST_FEATURE_DIM).to(device)
    fft_model = FFTMLP(FFT_FEATURE_DIM).to(device)
    cnn_model = SmallFireCNN().to(device)
    load_checkpoint(hist_model, args.hist_checkpoint, device)
    load_checkpoint(fft_model, args.fft_checkpoint, device)
    load_checkpoint(cnn_model, args.cnn_checkpoint, device)
    hist_model.eval()
    fft_model.eval()
    cnn_model.eval()

    moe = ExpertMoE(num_classes=len(CLASSES)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(moe.parameters(), lr=args.lr, weight_decay=1e-4)

    wandb_run = maybe_init_wandb(
        args.use_wandb,
        args.wandb_project,
        args.run_name or "moe_gate",
        {
            "model": "ExpertMoE",
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
        },
    )

    best_acc = 0.0
    artifact_path = args.artifacts_dir / "moe_gating.pth"
    args.artifacts_dir.mkdir(parents=True, exist_ok=True)

    def run_loader(loader, split: str, optimizer_to_use=None):
        if optimizer_to_use:
            moe.train()
        else:
            moe.eval()
        total_loss = 0.0
        total_correct = 0
        total_examples = 0
        for images, labels, paths in loader:
            hist_feats, fft_feats = compute_feature_batches(paths, cache_dir, split=split)
            images = images.to(device)
            labels = labels.to(device)
            hist_feats = hist_feats.to(device)
            fft_feats = fft_feats.to(device)
            with torch.no_grad():
                hist_logits = hist_model(hist_feats)
                fft_logits = fft_model(fft_feats)
                cnn_logits = cnn_model(images)
            expert_logits = torch.stack([hist_logits, fft_logits, cnn_logits], dim=1).detach()
            confidences = torch.stack(
                [
                    confidence_from_logits(hist_logits),
                    confidence_from_logits(fft_logits),
                    confidence_from_logits(cnn_logits),
                ],
                dim=1,
            ).detach()
            if optimizer_to_use:
                optimizer_to_use.zero_grad()
                logits = moe(expert_logits, confidences=confidences)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer_to_use.step()
            else:
                with torch.no_grad():
                    logits = moe(expert_logits, confidences=confidences)
                    loss = criterion(logits, labels)
            preds = torch.argmax(logits, dim=1)
            total_loss += loss.item() * labels.size(0)
            total_correct += (preds == labels).sum().item()
            total_examples += labels.size(0)
        avg_loss = total_loss / max(total_examples, 1)
        acc = total_correct / max(total_examples, 1)
        return avg_loss, acc

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_loader(train_loader, "train", optimizer)
        val_loss, val_acc = run_loader(val_loader, "val", optimizer_to_use=None)
        if wandb_run:
            wandb_run.log(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                }
            )
        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(
                {
                    "model": "moe",
                    "state_dict": moe.state_dict(),
                    "val_acc": val_acc,
                    "config": vars(args),
                },
                artifact_path,
            )
        print(
            f"[Epoch {epoch:03d}] train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

    if wandb_run:
        wandb_run.finish()


if __name__ == "__main__":
    main()
