from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from config_utils import parse_args_with_config
from datasets import CLASSES, ImageDataset
from features import FFT_FEATURE_DIM, HIST_FEATURE_DIM, load_fft_features, load_hist_features
from models import ExpertMoE, FFTMLP, HistMLP, SmallFireCNN
from utils import confidence_from_logits

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA_ROOT = BASE_DIR / "dataset"
DEFAULT_CACHE_DIR = BASE_DIR / ".feature_cache"
DEFAULT_ARTIFACTS = BASE_DIR / "artifacts"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run ablation experiments on validation split.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--hist-checkpoint", type=Path, default=DEFAULT_ARTIFACTS / "hist_expert.pth")
    parser.add_argument("--fft-checkpoint", type=Path, default=DEFAULT_ARTIFACTS / "fft_expert.pth")
    parser.add_argument("--cnn-checkpoint", type=Path, default=DEFAULT_ARTIFACTS / "cnn_expert.pth")
    parser.add_argument("--moe-checkpoint", type=Path, default=DEFAULT_ARTIFACTS / "moe_gating.pth")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--output", type=Path, default=BASE_DIR / "ablation.json")
    return parser


def parse_args() -> argparse.Namespace:
    parser = build_parser()
    return parse_args_with_config(parser)


def load_checkpoint(model: torch.nn.Module, path: Path, device: torch.device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
        ]
    )
    dataset = ImageDataset(args.data_root / "val", transform=transform, return_path=True)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    hist_model = HistMLP(HIST_FEATURE_DIM).to(device)
    fft_model = FFTMLP(FFT_FEATURE_DIM).to(device)
    cnn_model = SmallFireCNN().to(device)
    load_checkpoint(hist_model, args.hist_checkpoint, device)
    load_checkpoint(fft_model, args.fft_checkpoint, device)
    load_checkpoint(cnn_model, args.cnn_checkpoint, device)
    hist_model.eval()
    fft_model.eval()
    cnn_model.eval()

    moe_model = None
    if args.moe_checkpoint.exists():
        moe_model = ExpertMoE(num_classes=len(CLASSES)).to(device)
        load_checkpoint(moe_model, args.moe_checkpoint, device)
        moe_model.eval()

    hist_logits_all = []
    fft_logits_all = []
    cnn_logits_all = []
    labels_all = []

    with torch.no_grad():
        for images, labels, paths in loader:
            hist_feats = torch.from_numpy(
                np.stack([load_hist_features(path, cache_dir=args.cache_dir, split="val") for path in paths])
            ).float()
            fft_feats = torch.from_numpy(
                np.stack([load_fft_features(path, cache_dir=args.cache_dir, split="val") for path in paths])
            ).float()
            images = images.to(device)
            labels = labels.to(device)
            hist_feats = hist_feats.to(device)
            fft_feats = fft_feats.to(device)
            hist_logits = hist_model(hist_feats)
            fft_logits = fft_model(fft_feats)
            cnn_logits = cnn_model(images)
            hist_logits_all.append(hist_logits.cpu())
            fft_logits_all.append(fft_logits.cpu())
            cnn_logits_all.append(cnn_logits.cpu())
            labels_all.append(labels.cpu())

    hist_logits = torch.cat(hist_logits_all)
    fft_logits = torch.cat(fft_logits_all)
    cnn_logits = torch.cat(cnn_logits_all)
    labels = torch.cat(labels_all)

    def accuracy(logits: torch.Tensor) -> float:
        preds = torch.argmax(logits, dim=1)
        return float((preds == labels).float().mean().item())

    results = {
        "hist_only": accuracy(hist_logits),
        "fft_only": accuracy(fft_logits),
        "cnn_only": accuracy(cnn_logits),
        "hist_fft_avg": accuracy((hist_logits + fft_logits) / 2.0),
        "hist_cnn_avg": accuracy((hist_logits + cnn_logits) / 2.0),
        "fft_cnn_avg": accuracy((fft_logits + cnn_logits) / 2.0),
        "all_avg": accuracy((hist_logits + fft_logits + cnn_logits) / 3.0),
    }

    if moe_model is not None:
        with torch.no_grad():
            expert_logits = torch.stack([hist_logits, fft_logits, cnn_logits], dim=1)
            confidences = torch.stack(
                [
                    confidence_from_logits(hist_logits),
                    confidence_from_logits(fft_logits),
                    confidence_from_logits(cnn_logits),
                ],
                dim=1,
            )
            moe_preds = moe_model(expert_logits.to(device), confidences.to(device)).cpu()
        results["moe_combiner"] = accuracy(moe_preds)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("Ablation results:")
    for name, acc in results.items():
        print(f"  {name:15s}: {acc:.4f}")


if __name__ == "__main__":
    main()
