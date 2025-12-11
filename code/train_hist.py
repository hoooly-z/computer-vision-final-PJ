from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config_utils import parse_args_with_config
from datasets import FeatureDataset, CLASSES
from features import HIST_FEATURE_DIM
from models import HistMLP
from utils import evaluate_model, log_images_to_wandb, maybe_init_wandb, run_epoch, save_checkpoint, set_seed, get_device


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA_ROOT = BASE_DIR / "dataset"
DEFAULT_CACHE_DIR = BASE_DIR / ".feature_cache"
DEFAULT_ARTIFACTS = BASE_DIR / "artifacts"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train histogram expert (Expert A).")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="fire-moe")
    parser.add_argument("--run-name", type=str, default=None)
    return parser


def parse_args() -> argparse.Namespace:
    parser = build_parser()
    return parse_args_with_config(parser)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device()

    train_dir = args.data_root / "train"
    val_dir = args.data_root / "val"
    train_dataset = FeatureDataset(train_dir, "hist", cache_dir=args.cache_dir, split="train")
    val_dataset = FeatureDataset(val_dir, "hist", cache_dir=args.cache_dir, split="val")

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

    model = HistMLP(HIST_FEATURE_DIM).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    wandb_run = maybe_init_wandb(
        args.use_wandb,
        args.wandb_project,
        args.run_name or "hist_expert",
        {
            "model": "HistMLP",
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
        },
    )
    if wandb_run:
        random_samples = random.sample(train_dataset.samples, min(6, len(train_dataset)))
        log_images_to_wandb(wandb_run, "hist_expert_samples", random_samples, CLASSES)

    best_acc = 0.0
    artifact_path = args.artifacts_dir / "hist_expert.pth"
    args.artifacts_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)
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
                    "model": "hist",
                    "state_dict": model.state_dict(),
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
